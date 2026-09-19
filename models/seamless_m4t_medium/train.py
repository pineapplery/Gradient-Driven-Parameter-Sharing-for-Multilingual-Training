
"""
训练脚本，用于微调 SeamlessM4T Medium 模型。
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json

from modeling_seamless_m4t_medium import SeamlessM4TMediumModel
from dataset import S2TDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1
) -> float:
    """
    训练一个 epoch。

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        gradient_accumulation_steps: 梯度累积步数

    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        # 将数据移动到设备
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        tgt_langs = batch["tgt_langs"]

        # 计算损失
        loss = 0.0
        for i in range(len(tgt_langs)):
            single_loss = model.compute_loss(
                input_features[i:i+1],
                labels[i:i+1],
                tgt_lang=tgt_langs[i]
            )
            loss += single_loss

        loss = loss / len(tgt_langs)

        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps

        # 更新参数
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 更新进度条
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    评估模型。

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将数据移动到设备
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            tgt_langs = batch["tgt_langs"]

            # 计算损失
            loss = 0.0
            for i in range(len(tgt_langs)):
                single_loss = model.compute_loss(
                    input_features[i:i+1],
                    labels[i:i+1],
                    tgt_lang=tgt_langs[i]
                )
                loss += single_loss

            loss = loss / len(tgt_langs)
            total_loss += loss.item()

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SeamlessM4T Medium model")
    parser.add_argument("--model_name", type=str, default="facebook/hf-seamless-m4t-medium",
                        help="Model name or path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory containing TSV files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=480000,
                        help="Maximum audio length")
    parser.add_argument("--max_text_length", type=int, default=256,
                        help="Maximum text length")
    parser.add_argument("--use_fbank", action="store_true",
                        help="Use pre-computed fbank features")
    parser.add_argument("--fbank_dir", type=str, default=None,
                        help="Directory containing fbank features")
    parser.add_argument("--freeze_encoder_layers", type=int, default=2,
                        help="Number of encoder layers to unfreeze")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--language_pairs", type=str, nargs="+",
                        default=["aeb_en", "bem_en", "est_en", "gle_en"],
                        help="Language pairs to train on")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载模型
    logger.info(f"Loading model from {args.model_name}")
    model = SeamlessM4TMediumModel(model_name=args.model_name)
    model.to(device)

    # 冻结参数
    logger.info(f"Freezing all parameters except last {args.freeze_encoder_layers} encoder layers")
    model.prepare_for_finetune(n=args.freeze_encoder_layers)

    # 打印可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 准备数据
    logger.info("Preparing datasets...")
    train_datasets = []
    val_datasets = []

    for lang_pair in args.language_pairs:
        # 训练集
        train_tsv = Path(args.data_dir) / f"train_{lang_pair}.tsv"
        if train_tsv.exists():
            train_dataset = S2TDataset(
                tsv_file=str(train_tsv),
                processor=model.processor,
                max_length=args.max_length,
                max_text_length=args.max_text_length,
                use_fbank=args.use_fbank,
                fbank_dir=args.fbank_dir
            )
            train_datasets.append(train_dataset)
            logger.info(f"Loaded training data for {lang_pair}: {len(train_dataset)} samples")
        else:
            logger.warning(f"Training TSV file not found: {train_tsv}")

        # 验证集
        val_tsv = Path(args.data_dir) / f"val_{lang_pair}.tsv"
        if val_tsv.exists():
            val_dataset = S2TDataset(
                tsv_file=str(val_tsv),
                processor=model.processor,
                max_length=args.max_length,
                max_text_length=args.max_text_length,
                use_fbank=args.use_fbank,
                fbank_dir=args.fbank_dir
            )
            val_datasets.append(val_dataset)
            logger.info(f"Loaded validation data for {lang_pair}: {len(val_dataset)} samples")
        else:
            logger.warning(f"Validation TSV file not found: {val_tsv}")

    # 合并数据集
    from torch.utils.data import ConcatDataset
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
        logger.info(f"Total training samples: {len(train_dataset)}")
    else:
        raise ValueError("No training data found!")

    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
        logger.info(f"Total validation samples: {len(val_dataset)}")
    else:
        logger.warning("No validation data found!")
        val_dataset = None

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
    else:
        val_dataloader = None

    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 设置学习率调度器
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_steps
    )

    # 训练循环
    logger.info("Starting training...")
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.num_epochs):
        # 训练
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        logger.info(f"Epoch {epoch}: Train loss = {train_loss:.4f}")

        # 评估
        if val_dataloader:
            val_loss = evaluate(
                model=model,
                dataloader=val_dataloader,
                device=device
            )
            logger.info(f"Epoch {epoch}: Val loss = {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

        # 保存定期检查点
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
