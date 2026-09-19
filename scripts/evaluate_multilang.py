
# !/usr/bin/env python3
"""
多语言推理评估脚本：支持aeb_en, bem_en, est_en, gle_en四种语言对
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from seamless_communication.inference.translator import Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
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

def load_model(
    model_path: str,
    device: torch.device
) -> Translator:
    """
    加载微调后的模型

    Args:
        model_path: 模型路径
        device: 设备

    Returns:
        Translator: 翻译器
    """
    logger.info(f"Loading model from {model_path}")

    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "seamlessM4T_medium")

    # 创建翻译器
    translator = Translator(
        model_name_or_card=model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        input_modality=None,
        output_modality=None,
    )

    # 加载模型权重
    model_state_dict = checkpoint["model"]
    translator.model.load_state_dict(model_state_dict)
    translator.model.eval()

    logger.info("Model loaded successfully")
    return translator

def evaluate_on_dataset(
    translator: Translator,
    manifest_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 1
) -> Tuple[float, List[dict]]:
    """
    在数据集上评估模型

    Args:
        translator: 翻译器
        manifest_path: manifest文件路径
        output_path: 输出文件路径
        device: 设备
        batch_size: 批次大小

    Returns:
        Tuple[float, List[dict]]: (平均损失, 预测结果列表)
    """
    logger.info(f"Evaluating on {manifest_path}")

    # 加载manifest
    with open(manifest_path, "r") as f:
        samples = [json.loads(line) for line in f]

    # 解析语言对
    parts = manifest_path.stem.split("_", 1)
    if len(parts) != 2:
        logger.error(f"Invalid manifest name: {manifest_path}")
        return 0.0, []

    split_type, lang_pair = parts
    src_lang, tgt_lang = lang_pair.split("_")
    src_lang_code = LANG_CODE_MAPPING.get(src_lang, src_lang)
    tgt_lang_code = LANG_CODE_MAPPING.get(tgt_lang, tgt_lang)

    logger.info(f"Language pair: {src_lang_code} -> {tgt_lang_code}")

    # 评估
    results = []
    total_loss = 0.0
    num_samples = 0

    for sample in tqdm(samples, desc=f"Evaluating {lang_pair}"):
        try:
            # 预测
            with torch.no_grad():
                texts, _ = translator.predict(
                    input=sample["source"]["audio_local_path"],
                    task_str="S2TT",
                    src_lang=src_lang_code,
                    tgt_lang=tgt_lang_code,
                )

            # 获取预测文本
            if texts and len(texts) > 0:
                pred_text = str(texts[0])
            else:
                pred_text = ""

            # 获取目标文本
            target_text = sample["target"]["text"]

            # 计算字符错误率（简化版）
            # 这里可以使用更复杂的指标如BLEU、WER等
            cer = calculate_cer(pred_text, target_text)

            # 记录结果
            result = {
                "id": sample["source"]["id"],
                "source_lang": src_lang_code,
                "target_lang": tgt_lang_code,
                "target_text": target_text,
                "pred_text": pred_text,
                "cer": cer,
            }
            results.append(result)

            total_loss += cer
            num_samples += 1

        except Exception as e:
            logger.error(f"Error processing sample {sample['source']['id']}: {e}")
            continue

    # 计算平均CER
    avg_cer = total_loss / num_samples if num_samples > 0 else 0.0

    # 保存结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Average CER: {avg_cer:.4f}")
    logger.info(f"Results saved to {output_path}")

    return avg_cer, results

def calculate_cer(pred: str, target: str) -> float:
    """
    计算字符错误率（Character Error Rate）

    Args:
        pred: 预测文本
        target: 目标文本

    Returns:
        float: CER值
    """
    # 简化版CER计算
    if not target:
        return 1.0 if pred else 0.0

    # 转换为小写
    pred = pred.lower().strip()
    target = target.lower().strip()

    # 计算编辑距离
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1  # substitution
                )

    return dp[m][n] / n

def main():
    parser = argparse.ArgumentParser(description="Multi-language evaluation script")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        required=True,
        help="Directory containing manifest files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载模型
    translator = load_model(args.model_path, device)

    # 语言对列表
    lang_pairs = ["aeb_en", "bem_en", "est_en", "gle_en"]
    splits = ["valid", "test"]

    # 评估每个语言对
    results_summary = {}
    for lang_pair in lang_pairs:
        for split in splits:
            # 查找manifest文件
            manifest_path = Path(args.manifest_dir) / f"{split}_{lang_pair}_manifest.json"

            if not manifest_path.exists():
                logger.warning(f"Manifest not found: {manifest_path}")
                continue

            # 评估
            output_path = output_dir / f"{split}_{lang_pair}_results.json"
            avg_cer, _ = evaluate_on_dataset(
                translator=translator,
                manifest_path=manifest_path,
                output_path=output_path,
                device=device,
                batch_size=args.batch_size
            )

            # 记录结果
            key = f"{split}_{lang_pair}"
            results_summary[key] = avg_cer

    # 保存汇总结果
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("Evaluation Summary:")
    logger.info("=" * 50)
    for key, value in results_summary.items():
        logger.info(f"{key}: CER = {value:.4f}")
    logger.info("=" * 50)
    logger.info(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
