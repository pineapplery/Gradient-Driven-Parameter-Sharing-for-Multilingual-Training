
# !/usr/bin/env python3
"""
合并多个语言的manifest文件，用于混合训练
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> List[dict]:
    """
    加载manifest文件

    Args:
        manifest_path: manifest文件路径

    Returns:
        List[dict]: 样本列表
    """
    logger.info(f"Loading manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        samples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(samples)} samples from {manifest_path.name}")
    return samples


def save_manifest(samples: List[dict], output_path: Path) -> None:
    """
    保存manifest文件

    Args:
        samples: 样本列表
        output_path: 输出文件路径
    """
    logger.info(f"Saving {len(samples)} samples to {output_path}")
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Manifest saved to {output_path}")


def merge_manifests(
    manifest_paths: List[Path],
    output_path: Path,
    shuffle: bool = True
) -> None:
    """
    合并多个manifest文件

    Args:
        manifest_paths: manifest文件路径列表
        output_path: 输出文件路径
        shuffle: 是否打乱顺序
    """
    all_samples = []

    # 加载所有manifest
    for manifest_path in manifest_paths:
        samples = load_manifest(manifest_path)
        all_samples.extend(samples)

    logger.info(f"Total samples after merging: {len(all_samples)}")

    # 打乱顺序
    if shuffle:
        import random
        random.shuffle(all_samples)
        logger.info("Samples shuffled")

    # 保存合并后的manifest
    save_manifest(all_samples, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple manifest files for mixed training"
    )
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        required=True,
        help="Directory containing manifest files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save merged manifests",
    )
    parser.add_argument(
        "--lang_pairs",
        type=str,
        default="aeb_eng,bem_eng,est_eng,gle_eng",
        help="Comma-separated list of language pairs to merge",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Do not shuffle the merged samples",
    )

    args = parser.parse_args()

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 解析语言对列表
    lang_pairs = [lp.strip() for lp in args.lang_pairs.split(",")]
    logger.info(f"Language pairs to merge: {lang_pairs}")

    # 合并训练集 -> train_all_manifest.json (naming matches the convention used
    # by every training command in this repo, e.g. finetune_multilang.py)
    train_manifests = [
        args.manifest_dir / f"train_{lang_pair}_manifest.json"
        for lang_pair in lang_pairs
    ]
    train_output = args.output_dir / "train_all_manifest.json"
    merge_manifests(train_manifests, train_output, shuffle=not args.no_shuffle)

    # 合并验证集 -> valid_all_manifest.json
    valid_manifests = [
        args.manifest_dir / f"valid_{lang_pair}_manifest.json"
        for lang_pair in lang_pairs
    ]
    valid_output = args.output_dir / "valid_all_manifest.json"
    merge_manifests(valid_manifests, valid_output, shuffle=not args.no_shuffle)

    # 合并测试集（如果存在）-> test_all_manifest.json
    test_manifests = []
    for lang_pair in lang_pairs:
        test_path = args.manifest_dir / f"test_{lang_pair}_manifest.json"
        if test_path.exists():
            test_manifests.append(test_path)

    if test_manifests:
        test_output = args.output_dir / "test_all_manifest.json"
        merge_manifests(test_manifests, test_output, shuffle=not args.no_shuffle)

    logger.info("All manifests merged successfully!")


if __name__ == "__main__":
    main()
