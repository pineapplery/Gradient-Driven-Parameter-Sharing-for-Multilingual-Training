
# !/usr/bin/env python3
"""
多语言微调脚本：支持aeb_eng, bem_eng, est_eng, gle_eng四种语言对
支持参数冻结和预计算的fbank特征
"""

import argparse
import logging
import os
from pathlib import Path

import torch

from seamless_communication.cli.m4t.finetune import dist_utils
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

# 导入自定义的数据加载器
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune_multilang")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-language finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help=("Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    parser.add_argument(
        "--use_fbank",
        action="store_true",
        help="Use pre-computed fbank features instead of raw audio",
    )
    parser.add_argument(
        "--freeze_encoder_except_last_n",
        type=int,
        default=0,  # 修复：默认为0，不冻结任何层
        help="Number of last encoder layers to keep trainable (freeze all others). 0 = no freezing",
    )
    parser.add_argument(
        "--freeze_all_except_encoder",
        action="store_true",
        help="Freeze all parameters except encoder",
    )
    
    # ===== 新增：灵活的训练模式选择 =====
    parser.add_argument(
        "--train_adapter_only",
        action="store_true",
        help="Train only adapter modules (speech_encoder.adaptor_layers), freeze all others",
    )
    parser.add_argument(
        "--train_encoder_layers",
        type=str,
        help="Specify encoder layers to train (e.g., '10,11' or '10-11'), freeze all others. "
             "Use comma-separated indices (0-based) or range notation.",
    )
    return parser


def parse_layer_range(layer_spec: str) -> list:
    """
    解析层范围字符串，返回层索引列表
    
    支持格式：
    - "10,11": 逗号分隔的层索引
    - "10-11": 范围表示法
    - "9,10,11": 多个层
    - "10-12,15": 混合格式
    
    Args:
        layer_spec: 层范围字符串
        
    Returns:
        list: 层索引列表（0-based）
    """
    layer_indices = []
    
    for part in layer_spec.split(','):
        part = part.strip()
        if '-' in part:
            # 范围格式："10-11"
            start, end = map(int, part.split('-'))
            layer_indices.extend(range(start, end + 1))
        else:
            # 单个层："10"
            layer_indices.append(int(part))
    
    return sorted(set(layer_indices))  # 去重并排序


def freeze_all_except_adapter(model):
    """
    冻结所有参数，只解冻adapter模块
    
    Args:
        model: UnitYModel
    """
    logger.info("Freezing all parameters except adapter modules")
    
    # 第一步：冻结所有参数
    logger.info("Step 1: Freezing all parameters")
    for param in model.parameters():
        param.requires_grad = False
    
    # 第二步：只解冻adapter相关参数
    encoder = model.speech_encoder
    unfrozen_count = 0
    
    if hasattr(encoder, "adaptor_layers"):
        logger.info("Step 2: Unfreezing adaptor_layers")
        for i, adaptor_layer in enumerate(encoder.adaptor_layers):
            logger.info(f"Unfreezing adaptor_layer {i}")
            for param in adaptor_layer.parameters():
                param.requires_grad = True
                unfrozen_count += 1
    else:
        logger.warning("No adaptor_layers found in speech_encoder")
    
    logger.info(f"Total unfrozen parameters: {unfrozen_count}")
    
    # 注意：以下模块都被冻结：
    # - speech_encoder.inner (所有encoder层)
    # - speech_encoder.proj1, proj2 (投影层)
    # - speech_encoder.layer_norm (层归一化)
    # - 所有其他模块（text_decoder等）


def freeze_all_except_specific_layers(model, layer_indices: list):
    """
    冻结所有参数，只解冻指定的encoder层
    
    Args:
        model: UnitYModel
        layer_indices: 要解冻的层索引列表（0-based）
    """
    logger.info(f"Freezing all parameters except encoder layers: {layer_indices}")
    
    # 第一步：冻结所有参数
    logger.info("Step 1: Freezing all parameters")
    for param in model.parameters():
        param.requires_grad = False
    
    # 第二步：只解冻指定的encoder层
    encoder = model.speech_encoder
    if hasattr(encoder, "inner") and hasattr(encoder.inner, "layers"):
        total_layers = len(encoder.inner.layers)
        logger.info(f"Total encoder layers: {total_layers}")
        
        # 验证层索引
        valid_indices = [i for i in layer_indices if 0 <= i < total_layers]
        invalid_indices = [i for i in layer_indices if i < 0 or i >= total_layers]
        
        if invalid_indices:
            logger.warning(f"Invalid layer indices (will be skipped): {invalid_indices}")
        
        if not valid_indices:
            raise ValueError(f"No valid layer indices found. Total layers: {total_layers}, requested: {layer_indices}")
        
        logger.info(f"Step 2: Unfreezing layers {valid_indices}")
        for layer_idx in valid_indices:
            layer = encoder.inner.layers[layer_idx]
            logger.info(f"Unfreezing encoder layer {layer_idx}")
            for param in layer.parameters():
                param.requires_grad = True
    else:
        raise ValueError("speech_encoder does not have attribute 'inner.layers'")
    
    # 注意：以下模块都被冻结：
    # - speech_encoder.inner的其他层
    # - speech_encoder.proj1, proj2 (投影层)
    # - speech_encoder.adaptor_layers (适配器层)
    # - speech_encoder.layer_norm (层归一化)
    # - 所有其他模块（text_decoder等）


def freeze_encoder_layers(model, n_last_layers: int):
    """
    冻结encoder中除最后n层外的所有层
    
    注意：speech_encoder是一个UnitYEncoderAdaptor，包含：
    - inner: 真正的w2v2 encoder（包含多层）
    - proj1, proj2: 投影层
    - adaptor_layers: 适配器层
    - layer_norm: 层归一化
    
    Args:
        model: UnitYModel
        n_last_layers: 最后n层保持可训练
    """
    logger.info(f"Freezing encoder except last {n_last_layers} layers")

    # 第一步：冻结所有参数
    logger.info("Step 1: Freezing all parameters")
    for param in model.parameters():
        param.requires_grad = False

    # 第二步：解冻speech_encoder.inner的最后n层
    encoder = model.speech_encoder
    if hasattr(encoder, "inner") and hasattr(encoder.inner, "layers"):
        total_layers = len(encoder.inner.layers)
        logger.info(f"Total encoder layers in inner: {total_layers}")
        logger.info(f"Unfreezing last {n_last_layers} layers: {total_layers-n_last_layers} to {total_layers-1}")

        # 解冻最后n层
        for idx, layer in enumerate(encoder.inner.layers[-n_last_layers:]):
            logger.info(f"Unfreezing encoder layer {total_layers-n_last_layers+idx}")
            for param in layer.parameters():
                param.requires_grad = True
    else:
        raise ValueError("speech_encoder does not have attribute 'inner.layers'")

    # 注意：以下模块都被冻结，不会参与训练：
    # - speech_encoder.inner的前(total_layers-n_last_layers)层
    # - speech_encoder.proj1, proj2 (投影层)
    # - speech_encoder.adaptor_layers (适配器层)
    # - speech_encoder.layer_norm (层归一化)
    # - 所有其他模块（text_encoder, text_decoder, t2u_model等）



def main() -> None:
    args = init_parser().parse_args()

    dist_utils.init_distributed([logger])
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16

    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)

    logger.info(f"Loading model: {args.model_name}")
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    assert model.target_vocab_info == text_tokenizer.vocab_info

    # 设置为SPEECH_TO_TEXT模式
    if model.t2u_model is not None:
        model.t2u_model = None

    if model.text_encoder is not None:
        model.text_encoder = None

    # 参数冻结策略选择
    training_modes_count = sum([
        bool(args.freeze_all_except_encoder),
        bool(args.freeze_encoder_except_last_n is not None and args.freeze_encoder_except_last_n > 0),
        bool(args.train_adapter_only),
        bool(args.train_encoder_layers),
    ])
    
    if training_modes_count > 1:
        raise ValueError("Only one training mode can be specified: "
                        "--freeze_all_except_encoder, --freeze_encoder_except_last_n, "
                        "--train_adapter_only, or --train_encoder_layers")
    
    if args.train_adapter_only:
        # 新功能：只训练adapter模块
        freeze_all_except_adapter(model)
        
    elif args.train_encoder_layers:
        # 新功能：训练指定的encoder层
        try:
            layer_indices = parse_layer_range(args.train_encoder_layers)
            freeze_all_except_specific_layers(model, layer_indices)
        except ValueError as e:
            logger.error(f"Invalid layer specification '{args.train_encoder_layers}': {e}")
            logger.error("Examples: '10,11' or '10-11' or '9,10,11' or '8-10,11'")
            raise
            
    elif args.freeze_all_except_encoder:
        # 原有功能：冻结除encoder外的所有参数
        logger.info("Freezing all parameters except encoder")
        for name, param in model.named_parameters():
            if not name.startswith("speech_encoder"):
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    elif args.freeze_encoder_except_last_n is not None and args.freeze_encoder_except_last_n > 0:
        # 原有功能：冻结encoder中除最后n层外的所有层
        freeze_encoder_layers(model, args.freeze_encoder_except_last_n)
    
    else:
        # 默认：不冻结任何参数
        logger.info("No freezing applied - all parameters trainable")

    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Put model on selected device
    model = model.to(torch.device(args.device))

    # 创建数据加载器
    batching_config = BatchingConfig(
        batch_size=args.batch_size,
        rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        max_audio_length_sec=15.0,
        float_dtype=float_dtype,
        use_fbank=args.use_fbank,
    )

    train_dataloader = UnitYDataLoaderWithFbank(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=batching_config,
        dataset_manifest_path=str(args.train_dataset),
        max_src_tokens_per_batch=args.max_src_tokens)

    eval_dataloader = UnitYDataLoaderWithFbank(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=batching_config,
        dataset_manifest_path=str(args.eval_dataset))

    # 导入训练器
    from seamless_communication.cli.m4t.finetune import trainer

    # 创建训练参数
    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=trainer.FinetuneMode.SPEECH_TO_TEXT,
        save_model_path=args.save_model_to,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
    )

    logger.info(f"Finetune Params: {finetune_params}")

    # 创建训练器
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=None)  # 我们已经手动处理了参数冻结

    # ========== 新增：定期保存所有阶段checkpoint ==========
    # Monkey-patch: 在每次eval后都保存一次checkpoint
    logger.info("=== 启用定期checkpoint保存功能 ===")
    logger.info(f"Checkpoint保存目录: {finetune_params.save_model_path.parent}")
    
    orig_eval_model = finetune._eval_model
    
    def eval_model_with_periodic_ckpt(n_batches: int = 100):
        # 调用原始的eval逻辑
        orig_eval_model(n_batches)
        
        # 在eval后保存一次checkpoint
        # 估算总steps（基于数据集大小和参数）
        current_step = finetune.update_idx
        total_epochs = finetune.params.max_epochs
        epoch_idx = finetune.epoch_idx
        
        # 计算进度百分比
        progress_pct = 100.0 * epoch_idx / total_epochs if total_epochs > 0 else 0
        
        # 构造checkpoint文件名（包含step和进度百分比）
        ckpt_dir = finetune_params.save_model_path.parent
        ckpt_name = finetune_params.save_model_path.stem
        ckpt_path = ckpt_dir / f"{ckpt_name}_step{current_step}_pct{progress_pct:.1f}.pt"
        
        # 保存当前模型
        torch.save({
            'model': finetune.model.state_dict(),
            'step': current_step,
            'epoch': epoch_idx,
            'progress_pct': progress_pct,
            'max_epochs': total_epochs,
        }, ckpt_path)
        
        logger.info(f"✓ [Checkpoint已保存] {ckpt_path.name} | 训练进度: {progress_pct:.1f}% (epoch {epoch_idx}/{total_epochs}, step {current_step})")
    
    finetune._eval_model = eval_model_with_periodic_ckpt
    logger.info("✓ 定期checkpoint保存功能已启用")
    logger.info("  每次评估时会保存一个checkpoint，文件名包含step和进度百分比")
    logger.info("  训练结束后，可根据best checkpoint的step挑选80%进度附近的checkpoint")
    
    # ========== 开始训练 ==========
    finetune.run()
    
    logger.info("\n=== 训练完成 ===")
    logger.info(f"Best checkpoint: {finetune_params.save_model_path}")
    logger.info(f"所有阶段checkpoint保存在: {finetune_params.save_model_path.parent}")
    logger.info("提示: 可根据日志找到best checkpoint的step，然后挑选对应80%进度的checkpoint用于二阶段训练")


if __name__ == "__main__":
    main()
