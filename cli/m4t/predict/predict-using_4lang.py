# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import numpy as np  # Add this import for handling .npy files
import pandas as pd  # Add this import for handling TSV files
import torch
import torchaudio
from fairseq2.generation import NGramRepeatBlockProcessor

from seamless_communication.inference import SequenceGeneratorOptions, Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["ASR", "S2ST", "S2TT"],
        help=(
            "* `ASR` -- automatic speech recognition (transcription);"
            "* `S2ST` -- speech to speech translation;"
            "* `S2TT` -- speech to text translation;"
        )
    )
    parser.add_argument(
        "--tgt_lang", type=str, help="Target language to translate/transcribe into."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language, only required if input is text.",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the generated audio.",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Base model name (`seamlessM4T_medium`, "
            "`seamlessM4T_large`, `seamlessM4T_v2_large`)"
        ),
        default="seamlessM4T_v2_large",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        help="Path to a local .pt checkpoint to load into the model (optional).",
        default=None,
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        help="Vocoder model name",
        default="vocoder_v2",
    )
    # Text generation args.
    parser.add_argument(
        "--text_generation_beam_size",
        type=int,
        help="Beam size for incremental text decoding.",
        default=5,
    )
    parser.add_argument(
        "--text_generation_max_len_a",
        type=int,
        help="`a` in `ax + b` for incremental text decoding.",
        default=1,
    )
    parser.add_argument(
        "--text_generation_max_len_b",
        type=int,
        help="`b` in `ax + b` for incremental text decoding.",
        default=200,
    )
    parser.add_argument(
        "--text_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental text decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        help="Size of ngram repeat block for both text & unit decoding.",
        default=4,
    )
    # Unit generation args.
    parser.add_argument(
        "--unit_generation_beam_size",
        type=int,
        help=(
            "Beam size for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=5,
    )
    parser.add_argument(
        "--unit_generation_max_len_a",
        type=int,
        help=(
            "`a` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=25,
    )
    parser.add_argument(
        "--unit_generation_max_len_b",
        type=int,
        help=(
            "`b` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=50,
    )
    parser.add_argument(
        "--unit_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental unit decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--unit_generation_ngram_filtering",
        type=bool,
        help=(
            "If True, removes consecutive repeated ngrams"
            "from the decoded unit output."
        ),
        default=False,
    )
    parser.add_argument(
        "--text_unk_blocking",
        type=bool,
        help=(
            "If True, set penalty of UNK to inf in text generator "
            "to block unk output."
        ),
        default=False,
    )
    return parser


def set_generation_opts(
    args: Namespace,
) -> Tuple[SequenceGeneratorOptions, SequenceGeneratorOptions]:
    # Set text, unit generation opts.
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=args.text_generation_beam_size,
        soft_max_seq_len=(
            args.text_generation_max_len_a,
            args.text_generation_max_len_b,
        ),
    )
    if args.text_unk_blocking:
        text_generation_opts.unk_penalty = torch.inf
    if args.text_generation_ngram_blocking:
        text_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )

    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=args.unit_generation_beam_size,
        soft_max_seq_len=(
            args.unit_generation_max_len_a,
            args.unit_generation_max_len_b,
        ),
    )
    if args.unit_generation_ngram_blocking:
        unit_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )
    return text_generation_opts, unit_generation_opts

#新增
def clean_text(text: str) -> str:
    """Clean and normalize predicted text for evaluation."""
    import re
    text = str(text).strip()
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    # Fix spacing between sentences
    text = re.sub(r'([.!?,;:])\s*([A-Z])', r'\1 \2', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4T inference on supported tasks using Translator."
    )
    parser.add_argument("input", type=str, help="Audio WAV file path, TSV file path, or text input.")

    parser = add_inference_arguments(parser)
    args = parser.parse_args()
    if not args.task or not args.tgt_lang:
        raise Exception(
            "Please provide required arguments for evaluation -  task, tgt_lang"
        )

    if args.task.upper() in {"S2ST", "T2ST"} and args.output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running inference on {device=} with {dtype=}.")

    translator = Translator(args.model_name, args.vocoder_name, device, dtype=dtype)

    # If a local checkpoint is provided, try to load it into the model.
    if getattr(args, 'load_checkpoint', None):
        ckpt_path = args.load_checkpoint
        logger.info(f"Loading checkpoint from {ckpt_path}")

        def _strip_prefix_if_present(state_dict, prefix="model."):
            # if keys start with prefix, strip it
            keys = list(state_dict.keys())
            if any(k.startswith(prefix) for k in keys):
                return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            return state_dict

        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw, dict) and "model" in raw:
                state = raw["model"]
            else:
                state = raw
            state = _strip_prefix_if_present(state, prefix="model.")
            load_res = translator.model.load_state_dict(state, strict=False)
            logger.info(
                f"Checkpoint loaded. Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}"
            )
        except Exception as e:
            logger.exception(f"Failed to load checkpoint {ckpt_path}: {e}")

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    # If the input is a TSV file, extract audio paths
    if args.input.endswith(".tsv"):
        logger.info("Detected TSV file input. Extracting audio paths.")
        tsv_data = pd.read_csv(args.input, sep="\t")
        audio_paths = tsv_data['audio'].tolist()
        logger.info(f"Extracted {len(audio_paths)} audio paths from TSV file.")

        # 准备保存结果的列表
        results = []

        for audio_path in audio_paths:
            if audio_path.endswith(".npy"):
                logger.info(f"Loading .npy file: {audio_path}")
                wav = np.load(audio_path)  # Load the .npy file
                print(f"[DEBUG] Loaded npy: {audio_path}, shape: {wav.shape}")
                if wav.ndim == 1:
                    wav = np.expand_dims(wav, axis=0)  # Add channel dimension if missing
                wav = torch.tensor(wav, dtype=torch.float32)  # Convert to torch tensor

                # 不需要扩展特征维度，模型期望80维的fbank特征
                if wav.shape[1] == 80:
                    logger.info(f"Using 80-dim fbank features as expected by model: {wav.shape}")
                # 转换为float16精度，与模型一致
                wav = wav.half()

                # 检查序列长度是否足够
                seq_len = wav.shape[0]
                min_kernel_size = 31  # 以模型最大卷积核为例
                if seq_len < min_kernel_size:
                    logger.warning(f"[SKIP] {audio_path} too short for model (seq_len={seq_len} < kernel_size={min_kernel_size}), skipping.")
                    continue

                sample_rate = 16_000  # Assume the sample rate is 16kHz for .npy files
            else:
                wav, sample_rate = torchaudio.load(audio_path)

            # Ensure the tensor has the correct shape (channels, samples)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # Add channel dimension if missing

            translator_input = torchaudio.functional.resample(
                wav, orig_freq=sample_rate, new_freq=16_000
            )


            print(f"[DEBUG] translator_input type: {type(translator_input)}, shape: {getattr(translator_input, 'shape', None)}")
            text_output, speech_output = translator.predict(
                translator_input,
                args.task,
                args.tgt_lang,
                src_lang=args.src_lang,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            )
            print(f"[DEBUG] translator_output text: {text_output}, speech: {type(speech_output)}")

            # 保存翻译结果到results列表
            cleaned_text = clean_text(str(text_output[0])) if text_output else ''
            results.append({
                'audio': audio_path,
                'text': cleaned_text
            })

            if speech_output is not None:
                logger.info(f"Saving translated audio in {args.tgt_lang}")
                output_audio_path = args.output_path / Path(audio_path).stem
                torchaudio.save(
                    f"{output_audio_path}.wav",
                    speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                    sample_rate=speech_output.sample_rate,
                )
            logger.info(f"Translated text in {args.tgt_lang}: {text_output[0]}")

        # 保存所有结果到TSV文件
        if results:
            logger.info(f"Saving {len(results)} results to {args.output_path}")
            output_df = pd.DataFrame(results)
            output_df.to_csv(args.output_path, sep="	", index=False)
    else:
        # If the input is audio or text
        if args.task.upper() in {"S2ST", "ASR", "S2TT"}:
            wav, sample_rate = torchaudio.load(args.input)
            translator_input = torchaudio.functional.resample(
                wav, orig_freq=sample_rate, new_freq=16_000
            )
        else:
            translator_input = args.input

        text_output, speech_output = translator.predict(
            translator_input,
            args.task,
            args.tgt_lang,
            src_lang=args.src_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
        )

        if speech_output is not None:
            logger.info(f"Saving translated audio in {args.tgt_lang}")
            torchaudio.save(
                args.output_path,
                speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                sample_rate=speech_output.sample_rate,
            )
        logger.info(f"Translated text in {args.tgt_lang}: {text_output[0]}")


if __name__ == "__main__":
    main()
