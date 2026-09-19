
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under license found in the
# MIT_LICENSE file in the root directory of this source tree.


import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from fairseq2.data.text import TextTokenEncoder
from fairseq2.models.nllb import NllbTokenizer
from torch import Tensor
from torch.nn.functional import pad as pad_tensor
from torch.utils.data import DataLoader

from seamless_communication.datasets.datatypes import LangPairSample
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenEncoder,
    UnitTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class SeqsBatch:
    src_tokens: Optional[Tensor]
    src_lengths: Optional[Tensor]
    target_tokens: Optional[Tensor]
    prev_output_tokens: Optional[Tensor]
    target_lengths: Optional[Tensor]

    def __del__(self) -> None:
        """Explicitly delete tensors
        to force GPU memory cleanup"""
        for tensor in [
            self.src_tokens,
            self.src_lengths,
            self.target_tokens,
            self.prev_output_tokens,
            self.target_lengths,
        ]:
            if tensor is not None:
                del tensor


@dataclass
class MultimodalSeqsBatch:
    speech_to_text: SeqsBatch
    text_to_units: SeqsBatch

    def __del__(self) -> None:
        del self.speech_to_text
        del self.text_to_units


@dataclass
class BatchingConfig:
    fbank_feats_pad_idx: int = 0
    """The pad index to use in fbanks batching."""

    batch_size: int = 5
    """Fixed batch size to use"""

    max_audio_length_sec: float = 15.0
    """ Drop samples with source audio sample length above this threshold."""

    rank: int = 0
    """The rank of this worker in the process group."""

    world_size: int = 1
    """The world size of the process group."""

    num_workers: int = 2
    """Parallelism in dataset preparation."""

    float_dtype: torch.dtype = torch.float16
    """Select between fp16/fp32 for float tensors """

    use_fbank: bool = False
    """Whether to use pre-computed fbank features."""


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)  # type: ignore


class UnitYDataLoaderWithFbank:
    SAMPLE_RATE = 16_000

    def __init__(
        self,
        text_tokenizer: NllbTokenizer,
        unit_tokenizer: UnitTokenizer,
        dataset_manifest_path: str,
        batching_config: BatchingConfig,
        max_src_tokens_per_batch: int = 100000
    ):
        self.text_tokenizer = text_tokenizer
        self.text_encoders_per_lang: Dict[str, TextTokenEncoder] = {}
        self.unit_tokenizer = unit_tokenizer
        self.unit_encoders_per_lang: Dict[str, UnitTokenEncoder] = {}
        self.batching_config = batching_config
        self.dataset = self._load_manifest(dataset_manifest_path)
        self.max_src_tokens_per_batch = max_src_tokens_per_batch

    def get_dataloader(self) -> DataLoader[SeqsBatch]:
        subset = split_dataset_by_node(
            self.dataset,
            rank=self.batching_config.rank,
            world_size=self.batching_config.world_size,
        )
        data_loader = DataLoader(
            dataset=subset,
            batch_size=self.batching_config.batch_size,
            shuffle=True,
            num_workers=self.batching_config.num_workers,
            collate_fn=self._prepare_batch,
            worker_init_fn=worker_init_fn,
        )
        return data_loader

    def __iter__(self) -> Iterable[MultimodalSeqsBatch]:
        return self.get_dataloader().__iter__()

    def _load_manifest(self, manifest_path: str) -> Dataset:
        """Load manifest file."""
        with open(manifest_path, "r") as f:
            data = [json.loads(line) for line in f]

        # 确保数据类型一致，避免Arrow转换错误
        for sample in data:
            # 确保source和target中的字段类型正确
            if "source" in sample:
                if "id" in sample["source"]:
                    sample["source"]["id"] = str(sample["source"]["id"])
                if "lang" in sample["source"]:
                    sample["source"]["lang"] = str(sample["source"]["lang"])
                if "text" in sample["source"]:
                    sample["source"]["text"] = str(sample["source"]["text"])
                if "audio_local_path" in sample["source"]:
                    sample["source"]["audio_local_path"] = str(sample["source"]["audio_local_path"])
                if "sampling_rate" in sample["source"] and sample["source"]["sampling_rate"] is not None:
                    sample["source"]["sampling_rate"] = int(sample["source"]["sampling_rate"])

            if "target" in sample:
                if "id" in sample["target"]:
                    sample["target"]["id"] = str(sample["target"]["id"])
                if "lang" in sample["target"]:
                    sample["target"]["lang"] = str(sample["target"]["lang"])
                if "text" in sample["target"]:
                    sample["target"]["text"] = str(sample["target"]["text"])

        return Dataset.from_list(data)

    def _get_source_fbank(self, sample: LangPairSample) -> Tensor:
        """Load pre-computed fbank features."""
        if self.batching_config.use_fbank:
            # Load pre-computed fbank features
            fbank_path = sample.source.audio_local_path
            if not fbank_path.endswith('.npy'):
                # If not .npy, try to find the corresponding .npy file
                fbank_path = str(Path(fbank_path).with_suffix('.npy'))

            fbank = np.load(fbank_path)
            fbank_tensor = torch.from_numpy(fbank).to(self.batching_config.float_dtype)
            return fbank_tensor
        else:
            # Original implementation: load audio and extract fbank
            import torchaudio
            from fairseq2.data.audio import WaveformToFbankConverter

            wav, sample_rate = torchaudio.load(sample.source.audio_local_path)
            assert (
                int(sample_rate) == self.SAMPLE_RATE
            ), f"sample != {self.SAMPLE_RATE}, please resample"
            assert len(wav.shape) in (1, 2)
            if len(wav.shape) == 1:
                wav = wav.unsqueeze(-1)
            elif wav.shape[0] <= 2:  # channel is first, should be second
                wav = wav.transpose(0, 1)

            fbank_extract_params = {
                "num_mel_bins": 80,
                "waveform_scale": 32768,
                "channel_last": True,
                "standardize": True,
                "device": torch.device("cpu"),
                "dtype": self.batching_config.float_dtype,
            }
            return WaveformToFbankConverter(**fbank_extract_params)(
                {
                    "waveform": wav,
                    "sample_rate": self.SAMPLE_RATE,
                }
            )["fbank"]

    def _get_tokenized_target_text(self, sample: LangPairSample) -> Tensor:
        """Expected sequence is [<eos>, <lang_tok> , ..text tokens.., <eos>]"""
        target_lang = sample.target.lang
        if target_lang not in self.text_encoders_per_lang:
            self.text_encoders_per_lang[target_lang] = (
                self.text_tokenizer.create_encoder(lang=target_lang, mode="target")
            )
        tokens = self.text_encoders_per_lang[target_lang](sample.target.text)
        eos_idx = self.text_tokenizer.vocab_info.eos_idx
        tokens = torch.concat([tokens, torch.LongTensor([eos_idx])])
        return tokens

    def _get_tokenized_units(self, sample: LangPairSample) -> Optional[Tensor]:
        """Expected sequence is [<eos>, <lang_tok> , ..unit tokens.., <eos>]"""
        if sample.target.units is None:
            return None
        target_lang = sample.target.lang
        if target_lang not in self.unit_encoders_per_lang:
            self.unit_encoders_per_lang[target_lang] = (
                self.unit_tokenizer.create_encoder(lang=target_lang)
            )
        tokens = self.unit_encoders_per_lang[target_lang](
            torch.LongTensor(sample.target.units).unsqueeze(0)
        )
        eos_idx = self.unit_tokenizer.vocab_info.eos_idx
        tokens = torch.concat([tokens.squeeze(0), torch.LongTensor([eos_idx])])
        return tokens

    def _batch_tensors(self, tensors: List[Tensor], pad_value: Any) -> Tensor:
        padding_size = max(tensor.shape[0] for tensor in tensors)
        dims = len(tensors[0].shape)
        padded_tensors = []
        for tensor in tensors:
            padding = [0] * 2 * dims
            padding[-1] = padding_size - tensor.shape[0]
            padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
        return torch.stack([tensor for tensor in padded_tensors], dim=0)

    def _is_long_src_audio(self, sample: LangPairSample) -> bool:
        """Check if audio is too long."""
        if self.batching_config.use_fbank:
            # Check fbank length instead of audio length
            fbank_path = sample.source.audio_local_path
            if not fbank_path.endswith('.npy'):
                fbank_path = str(Path(fbank_path).with_suffix('.npy'))

            fbank = np.load(fbank_path)
            length_s = fbank.shape[0] / 100.0  # Approximate: 100 frames per second
            return length_s > self.batching_config.max_audio_length_sec
        else:
            # Original implementation: check audio length
            import torchaudio
            try:
                wav, sample_rate = torchaudio.load(sample.source.audio_local_path)
                length_s: float = max(wav.shape) / sample_rate
                return length_s > self.batching_config.max_audio_length_sec
            except:
                logger.exception(f"Failed to load sample path: {sample.source.audio_local_path}")
                return True

    def _drop_overflow_samples(
        self, samples_with_fbanks: List[Tuple[LangPairSample, torch.Tensor]]
    ) -> List[Tuple[LangPairSample, torch.Tensor]]:
        # filter by src_tokens length (reverse)
        samples_with_fbanks = sorted(
            samples_with_fbanks, key=lambda sb: -sb[1].shape[0]
        )
        bwd = samples_with_fbanks[0][1].shape[0]
        max_samples_for_batch = max(1, self.max_src_tokens_per_batch // bwd)
        if max_samples_for_batch < len(samples_with_fbanks):
            samples_with_fbanks = samples_with_fbanks[:max_samples_for_batch]
        return samples_with_fbanks

    def _prepare_batch(self, raw_samples: List[Dict[str, Any]]) -> MultimodalSeqsBatch:
        samples = [LangPairSample.from_json(sample) for sample in raw_samples]
        # input speech

        #  - filter long audio samples
        filtered_samples = [
            sample for sample in samples if not self._is_long_src_audio(sample)
        ]
        samples = (
            filtered_samples if filtered_samples else [samples[0]]
        )  # keep at least one sample
        with_fbanks = [(sample, self._get_source_fbank(sample)) for sample in samples]
        #  - filter NaNs in fbanks
        filtered = [
            (sample, fbank)
            for sample, fbank in with_fbanks
            if not fbank.isnan().any().item()
        ]
        filtered = self._drop_overflow_samples(filtered)

        samples = [sample for sample, _ in filtered]
        src_tokens_list = [src_tokens for _, src_tokens in filtered]
        assert len(samples) > 0
        src_tokens = self._batch_tensors(
            src_tokens_list, pad_value=self.batching_config.fbank_feats_pad_idx
        ).to(self.batching_config.float_dtype)
        src_lengths = torch.LongTensor(
            [src_tokens.shape[0] for src_tokens in src_tokens_list]
        )

        # output text
        text_tokens_list = [
            self._get_tokenized_target_text(sample) for sample in samples
        ]
        text_pad_idx = self.text_tokenizer.vocab_info.pad_idx
        prev_outputs_tokens = self._batch_tensors(
            [tokens[:-1] for tokens in text_tokens_list], pad_value=text_pad_idx
        )
        target_tokens = self._batch_tensors(
            [tokens[1:] for tokens in text_tokens_list], pad_value=text_pad_idx
        )
        tokens_lengths = torch.LongTensor(
            [tokens.shape[0] - 1 for tokens in text_tokens_list]
        )
        # output units
        units_list_raw = [self._get_tokenized_units(sample) for sample in samples]
        units_list = [u for u in units_list_raw if u is not None]
        if units_list:
            units_pad_idx = self.unit_tokenizer.vocab_info.pad_idx
            prev_output_units = self._batch_tensors(
                [tokens[:-1] for tokens in units_list], pad_value=units_pad_idx
            )
            target_units = self._batch_tensors(
                [tokens[1:] for tokens in units_list], pad_value=units_pad_idx
            )
            units_lengths = torch.LongTensor(
                [tokens.shape[0] - 1 for tokens in units_list]
            )
            text_to_units_batch = SeqsBatch(
                src_tokens=None,
                src_lengths=None,
                target_tokens=target_units,
                prev_output_tokens=prev_output_units,
                target_lengths=units_lengths,
            )
        else:
            text_to_units_batch = SeqsBatch(
                src_tokens=None,
                src_lengths=None,
                target_tokens=None,
                prev_output_tokens=None,
                target_lengths=None,
            )

        speech_to_text_batch = SeqsBatch(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            target_tokens=target_tokens,
            prev_output_tokens=prev_outputs_tokens,
            target_lengths=tokens_lengths,
        )

        return MultimodalSeqsBatch(
            speech_to_text=speech_to_text_batch,
            text_to_units=text_to_units_batch,
        )
