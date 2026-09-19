
# !/usr/bin/env python3
"""
数据准备脚本：将TSV数据转换为manifest格式
支持多种语言对：aeb_eng, bem_eng, est_eng, gle_eng
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 语言代码映射
LANG_CODE_MAPPING = {
    "aeb": "aeb",
    "bem": "bem",
    "est": "est",
    "gle": "gle",
    "eng": "eng"
}

# 语言token映射
LANG_TOKEN_MAPPING = {
    "aeb": 256005,
    "bem": 256025,
    "est": 256049,
    "gle": 256061,
    "eng": 256047
}

def create_manifest_from_tsv(
    tsv_path: Path,
    output_path: Path,
    use_fbank: bool = False,
    fbank_dir: Optional[Path] = None
) -> None:
    """
    从TSV文件创建manifest

    Args:
        tsv_path: TSV文件路径
        output_path: 输出manifest文件路径
        use_fbank: 是否使用预计算的fbank特征
        fbank_dir: fbank特征目录
    """
    logger.info(f"Processing TSV: {tsv_path}")

    # 读取TSV文件
    df = pd.read_csv(tsv_path, sep="\t")

    # 解析语言对，处理可能存在的 `_npy` 后缀（例如: train_aeb_en_npy.tsv）
    parts = tsv_path.stem.split("_", 1)
    if len(parts) != 2:
        logger.warning(f"Unexpected TSV name format: {tsv_path.name}")
        return

    split_type, lang_pair = parts
    # 如果文件名以 `_npy` 结尾（由预计算特征生成的 TSV），去掉该后缀
    if lang_pair.endswith("_npy"):
        lang_pair = lang_pair[:-4]

    # 现在 lang_pair 应该形如 `aeb_en`，仅按第一个下划线分割源/目标语言
    if "_" not in lang_pair:
        logger.warning(f"Unexpected language pair format in filename: {tsv_path.name}")
        return
    src_lang, tgt_lang = lang_pair.split("_", 1)

    # 验证语言代码
    if src_lang not in LANG_CODE_MAPPING or tgt_lang not in LANG_CODE_MAPPING:
        logger.error(f"Unsupported language pair: {src_lang}_{tgt_lang}")
        return

    src_lang_code = LANG_CODE_MAPPING[src_lang]
    tgt_lang_code = LANG_CODE_MAPPING[tgt_lang]

    logger.info(f"Language pair: {src_lang_code} -> {tgt_lang_code}")

    # 创建manifest
    manifest_data = []
    for idx, row in df.iterrows():
        # 确定音频路径
        if use_fbank and fbank_dir is not None:
            # 使用预计算的fbank特征
            audio_path = row["audio"]
            if not Path(audio_path).exists():
                logger.warning(f"Fbank file not found: {audio_path}")
                continue
        else:
            # 使用原始音频文件
            audio_path = row["audio"]
            if not Path(audio_path).exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue

        # 创建样本
        sample = {
            "source": {
                "id": row.get("id", f"{idx}"),
                "text": "",  # 音频不需要源文本
                "lang": src_lang_code,
                "audio_local_path": audio_path,
                "sampling_rate": 16000 if use_fbank else None,
            },
            "target": {
                "id": row.get("id", f"{idx}"),
                "text": row.get("tgt_text", ""),
                "lang": tgt_lang_code,
            }
        }

        manifest_data.append(sample)

    # 写出manifest
    with open(output_path, "w") as f:
        for sample in manifest_data:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Saved manifest with {len(manifest_data)} samples to {output_path}")

def prepare_all_manifests(
    base_dir: Path,
    use_fbank: bool = False,
    fbank_dir: Optional[Path] = None
) -> None:
    """
    准备所有语言的manifest

    Args:
        base_dir: 基础目录
        use_fbank: 是否使用预计算的fbank特征
        fbank_dir: fbank特征目录
    """
    # 语言对列表
    lang_pairs = ["aeb_eng", "bem_eng", "est_eng", "gle_eng"]
    splits = ["train", "valid", "test"]

    # 创建输出目录
    manifest_dir = base_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 为每个语言对和split创建manifest
    for lang_pair in lang_pairs:
        for split in splits:
            # 查找TSV文件
            tsv_pattern = f"{split}_{lang_pair}.tsv"
            tsv_pattern_npy = f"{split}_{lang_pair}_npy.tsv"

            # 优先使用npy版本的TSV
            tsv_file = None
            for tsv_dir in [base_dir]:
                tsv_path = tsv_dir / tsv_pattern_npy
                if tsv_path.exists():
                    tsv_file = tsv_path
                    use_npy = True
                    break

                tsv_path = tsv_dir / tsv_pattern
                if tsv_path.exists():
                    tsv_file = tsv_path
                    use_npy = False
                    break

            if tsv_file is None:
                logger.warning(f"TSV file not found for {split}_{lang_pair}")
                continue

            # 确定输出路径
            output_name = f"{split}_{lang_pair}_manifest.json"
            output_path = manifest_dir / output_name

            # 创建manifest
            create_manifest_from_tsv(
                tsv_path=tsv_file,
                output_path=output_path,
                use_fbank=use_fbank and use_npy,
                fbank_dir=fbank_dir
            )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert per-language TSVs into per-language manifest.json files"
    )
    parser.add_argument(
        "--base_dir", type=Path, required=True,
        help="Directory containing the {split}_{lang}_eng[_npy].tsv files; "
             "manifests are written to {base_dir}/manifests/",
    )
    parser.add_argument(
        "--use_fbank", action="store_true", default=False,
        help="Use precomputed fbank .npy features (expects *_npy.tsv files)",
    )
    parser.add_argument(
        "--fbank_dir", type=Path, default=None,
        help="Directory of precomputed fbank features (only used for bookkeeping; "
             "actual fbank paths are read from the *_npy.tsv audio column)",
    )
    args = parser.parse_args()

    fbank_dir = args.fbank_dir if args.fbank_dir is not None else (args.base_dir / "fbank_features")

    prepare_all_manifests(
        base_dir=args.base_dir,
        use_fbank=args.use_fbank,
        fbank_dir=fbank_dir,
    )

    logger.info("All manifests prepared successfully!")

if __name__ == "__main__":
    main()
