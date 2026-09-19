
#!/usr/bin/env python3
"""
多语言微调脚本：支持共享和独立权重训练

正确的逻辑：
1. freeze_encoder_except_last_n: encoder最后N层不冻结（可训练）
2. num_independent_layers: 从未冻结层中，从顶层往前数N层为独立层
3. num_shared_layers: 从未冻结层中，从底层往后数N层为共享训练层

示例：假设encoder有6层
- freeze_encoder_except_last_n = 2 (最后2层不冻结：层4、层5)
- num_independent_layers = 2 (从层4、5中，从顶层往前数2层：层4、层5都独立)
- num_shared_layers = 0 (从层4、5中，从底层往后数0层：没有共享训练层)

结果：
- 层0-3: 冻结（使用预训练权重）
- 层4: group1独立 / group2独立
- 层5: group1独立 / group2独立
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
from seamless_communication.models.unity.multilang_encoder import MultiLangEncoderWrapper

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune_multilang_main")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-language finetuning script with shared/independent weights"
    )
    parser.add_argument("--train_dataset", type=Path, required=True)
    parser.add_argument("--eval_dataset", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="seamlessM4T_medium")
    parser.add_argument("--save_model_to", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2343)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=5000)
    parser.add_argument("--max_src_tokens", type=int, default=7000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_fbank", action="store_true")
    parser.add_argument("--freeze_encoder_except_last_n", type=int, default=2)
    parser.add_argument("--num_independent_layers", type=int, default=2)
    parser.add_argument("--num_shared_layers", type=int, default=0)
    parser.add_argument("--lang_groups", type=str, default=None)
    parser.add_argument("--train_group", type=str, default=None)
    return parser


def parse_lang_groups(lang_groups_str: str) -> dict:
    if lang_groups_str is None:
        return None
    groups = {}
    for group_str in lang_groups_str.split(';'):
        group_name, langs = group_str.split(':')
        groups[group_name] = langs.split(',')
    return groups


def setup_layer_freezing(model, freeze_encoder_except_last_n, num_independent_layers,
                      num_shared_layers, lang_groups, train_group):
    """
    设置模型参数的冻结/解冻

    关键修改：
    1. 首先冻结所有参数
    2. 只解冻speech_encoder的最后N层
    3. 确保不解冻embedding、projection等层
    """
    # 第一步：冻结所有参数
    logger.info("Step 1: Freezing all parameters")
    for param in model.parameters():
        param.requires_grad = False

    speech_encoder = model.speech_encoder

    # 检查是否是MultiLangEncoderWrapper
    if isinstance(speech_encoder, MultiLangEncoderWrapper):
        # MultiLangEncoderWrapper已经处理了层的冻结/解冻
        logger.info("Using MultiLangEncoderWrapper - layer freezing already configured")

        # 只需要冻结其他组的独立层
        if lang_groups is not None and train_group is not None:
            logger.info(f"Training only group: {train_group}")
            # 冻结其他组的独立层
            for group_name in lang_groups.keys():
                if group_name != train_group:
                    speech_encoder.freeze_group_layers(group_name)

        # 关键修复：只解冻指定组的独立层
        logger.info("Step 2: Unfreezing only specified group's independent layers")

        # 解冻指定组的独立层
        if train_group is not None and train_group in speech_encoder.independent_layers:
            independent_layers = speech_encoder.independent_layers[train_group]
            logger.info(f"Unfreezing {len(independent_layers)} independent layers for group: {train_group}")
            for layer in independent_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        # 统计可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # 打印可训练参数的详细信息
        logger.info("Trainable parameters details:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: {param.numel():,}")

        return

    # 对于普通编码器，使用原来的逻辑
    all_layers = get_encoder_layers(speech_encoder)
    total_layers = len(all_layers)

    frozen_count = total_layers - freeze_encoder_except_last_n
    unfrozen_start_idx = frozen_count
    unfrozen_end_idx = total_layers

    independent_end_idx = unfrozen_end_idx
    independent_start_idx = independent_end_idx - num_independent_layers

    shared_start_idx = unfrozen_start_idx
    shared_end_idx = shared_start_idx + num_shared_layers

    logger.info(f"Total encoder layers: {total_layers}")
    logger.info(f"Frozen layers: 0 to {frozen_count-1}")
    logger.info(f"Unfrozen layers: {unfrozen_start_idx} to {unfrozen_end_idx-1}")
    logger.info(f"  - Independent layers (group-specific): {independent_start_idx} to {independent_end_idx-1}")
    logger.info(f"  - Shared training layers: {shared_start_idx} to {shared_end_idx-1}")

    if shared_end_idx > independent_start_idx:
        raise ValueError(f"Invalid layer configuration")

    for i in range(unfrozen_start_idx, total_layers):
        for param in all_layers[i].parameters():
            param.requires_grad = True

    if lang_groups is not None and train_group is not None:
        logger.info(f"Training only group: {train_group}")
        if hasattr(speech_encoder, 'freeze_group_layers'):
            for group_name in lang_groups.keys():
                if group_name != train_group:
                    speech_encoder.freeze_group_layers(group_name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")


def get_encoder_layers(speech_encoder):
    """
    获取编码器的所有层

    对于UnitYEncoderAdaptor：
    - inner.layers: w2v2 encoder的层（底层）
    - adaptor_layers: 适配器层（顶层）

    返回所有层的列表：inner.layers + adaptor_layers
    """
    if hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
        # UnitYEncoderAdaptor的情况
        inner_layers = list(speech_encoder.inner.layers)
        adaptor_layers = list(speech_encoder.adaptor_layers)
        return inner_layers + adaptor_layers
    elif hasattr(speech_encoder, 'layers'):
        # 普通TransformerEncoder的情况
        return list(speech_encoder.layers)
    else:
        raise ValueError(f"Cannot find encoder layers in speech_encoder: {type(speech_encoder)}")


def main():
    args = init_parser().parse_args()
    dist_utils.init_distributed([logger])
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16

    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)

    logger.info(f"Loading model: {args.model_name}")
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    assert model.target_vocab_info == text_tokenizer.vocab_info

    if model.t2u_model is not None:
        model.t2u_model = None
    if model.text_encoder is not None:
        model.text_encoder = None

    lang_groups = parse_lang_groups(args.lang_groups)

    if lang_groups is not None:
        logger.info(f"Using language groups: {lang_groups}")

        speech_encoder = model.speech_encoder

        # 只获取inner_layers，不包含adaptor_layers
        if hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
            inner_layers = list(speech_encoder.inner.layers)
            total_layers = len(inner_layers)
        elif hasattr(speech_encoder, 'layers'):
            inner_layers = list(speech_encoder.layers)
            total_layers = len(inner_layers)
        else:
            raise ValueError(f"Cannot find encoder layers in speech_encoder: {type(speech_encoder)}")

        logger.info(f"Total encoder layers: {total_layers}")

        multilang_encoder = MultiLangEncoderWrapper(
            base_encoder=speech_encoder,
            num_frozen_layers=total_layers - args.freeze_encoder_except_last_n,
            num_shared_training_layers=args.num_shared_layers,
            num_independent_layers=args.num_independent_layers,
            lang_groups=lang_groups,
            default_group=args.train_group,  # 设置默认组
        )

        model.speech_encoder = multilang_encoder
        setup_layer_freezing(model, args.freeze_encoder_except_last_n,
                          args.num_independent_layers, args.num_shared_layers,
                          lang_groups, args.train_group)
    else:
        logger.info("No language groups specified, using unified training")
        setup_layer_freezing(model, args.freeze_encoder_except_last_n,
                          args.num_independent_layers, args.num_shared_layers,
                          None, None)

    model = model.to(torch.device(args.device))

    # 创建数据加载器
    from dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig

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

    from seamless_communication.cli.m4t.finetune import trainer

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

    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=None)

    finetune.run()


if __name__ == "__main__":
    main()
