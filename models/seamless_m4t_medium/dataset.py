
"""
数据加载器，用于加载 S2T 任务的训练数据。
"""
import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class S2TDataset(Dataset):
    """
    S2T 数据集类，用于加载音频和对应的翻译文本。
    """

    def __init__(
        self,
        tsv_file: str,
        processor,
        max_length: int = 480000,  # 最大音频长度 (16kHz * 30s)
        max_text_length: int = 256,  # 最大文本长度
        use_fbank: bool = False,
        fbank_dir: Optional[str] = None
    ):
        """
        初始化数据集。

        Args:
            tsv_file: TSV 文件路径
            processor: SeamlessM4T 的处理器
            max_length: 最大音频长度
            max_text_length: 最大文本长度
            use_fbank: 是否使用预计算的 fbank 特征
            fbank_dir: fbank 特征目录
        """
        self.tsv_file = tsv_file
        self.processor = processor
        self.max_length = max_length
        self.max_text_length = max_text_length
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        # 读取 TSV 文件
        self.data = pd.read_csv(tsv_file, sep="\t")

        # 从文件名提取语言对
        filename = Path(tsv_file).stem
        parts = filename.split("_")
        if len(parts) >= 2:
            self.src_lang = parts[1]
            self.tgt_lang = "eng"
        else:
            raise ValueError(f"Invalid TSV filename: {filename}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            包含输入特征和标签的字典
        """
        row = self.data.iloc[idx]
        audio_path = row['audio']
        tgt_text = row['tgt_text']

        # 加载音频或 fbank 特征
        if self.use_fbank:
            # 加载预计算的 fbank 特征
            audio_filename = Path(audio_path).stem
            fbank_path = Path(self.fbank_dir) / f"{audio_filename}.npy"
            if not fbank_path.exists():
                raise FileNotFoundError(f"Fbank file not found: {fbank_path}")

            input_features = np.load(fbank_path)
            input_features = torch.from_numpy(input_features).float()
        else:
            # 加载原始音频
            waveform, sample_rate = torchaudio.load(audio_path)

            # 重采样到 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)

            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 截断或填充音频
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]

            # 使用 processor 处理音频
            inputs = self.processor(
                audio=waveform.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_features = inputs["input_features"].squeeze(0)

        # 处理目标文本
        text_inputs = self.processor(
            text=tgt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length
        )
        labels = text_inputs["input_ids"].squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "audio_path": audio_path
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义批处理函数，用于将样本组合成批次。

    Args:
        batch: 样本列表

    Returns:
        批次数据
    """
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]
    src_langs = [item["src_lang"] for item in batch]
    tgt_langs = [item["tgt_lang"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    # 填充输入特征
    input_features = torch.nn.utils.rnn.pad_sequence(
        input_features, batch_first=True, padding_value=0.0
    )

    # 填充标签
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_features": input_features,
        "labels": labels,
        "src_langs": src_langs,
        "tgt_langs": tgt_langs,
        "audio_paths": audio_paths
    }
