
#!/usr/bin/env python3
"""
合并多语言模型权重

将group1和group2训练好的独立层权重合并到一个模型中
"""

import torch
import argparse
from pathlib import Path


def merge_models(group1_path, group2_path, output_path, lang_groups):
    """
    合并两个组的模型权重

    Args:
        group1_path: group1模型路径
        group2_path: group2模型路径
        output_path: 输出模型路径
        lang_groups: 语言分组配置
    """
    # 加载两个组的模型
    print(f"Loading group1 model from {group1_path}")
    group1_model = torch.load(group1_path, map_location='cpu')

    print(f"Loading group2 model from {group2_path}")
    group2_model = torch.load(group2_path, map_location='cpu')

    # 创建合并后的模型
    merged_model = {
        'model': {},
        'model_name': group1_model.get('model_name', 'seamlessM4T_medium'),
        'lang_groups': lang_groups,
        'layer_config': {
            'freeze_encoder_except_last_n': 2,
            'num_independent_layers': 2,
            'num_shared_layers': 0
        }
    }

    # 统计信息
    group1_keys = set(group1_model['model'].keys())
    group2_keys = set(group2_model['model'].keys())

    # 分析权重key的结构
    print(f"Group1 model has {len(group1_keys)} keys")
    print(f"Group2 model has {len(group2_keys)} keys")

    # 找出所有包含speech_encoder的key
    group1_speech_encoder_keys = [k for k in group1_keys if 'speech_encoder' in k]
    group2_speech_encoder_keys = [k for k in group2_keys if 'speech_encoder' in k]

    print(f"Group1 speech_encoder keys: {len(group1_speech_encoder_keys)}")
    print(f"Group2 speech_encoder keys: {len(group2_speech_encoder_keys)}")

    # 合并权重
    print("Merging weights...")

    # 1. 处理speech_encoder之外的权重（使用group1的）
    for key in group1_model['model']:
        if 'speech_encoder' not in key:
            merged_model['model'][key] = group1_model['model'][key]
            print(f"  Copied non-speech_encoder: {key}")

    # 2. 处理speech_encoder的权重
    for key in group1_model['model']:
        if 'speech_encoder' not in key:
            continue

        # 冻结层（使用group1的）
        if 'frozen_layers' in key:
            merged_model['model'][key] = group1_model['model'][key]
            print(f"  Copied frozen layer: {key}")

        # 共享训练层（使用group1的）
        elif 'shared_training_layers' in key:
            merged_model['model'][key] = group1_model['model'][key]
            print(f"  Copied shared training layer: {key}")

        # adaptor层（使用group1的）
        elif 'adaptor_layers' in key:
            merged_model['model'][key] = group1_model['model'][key]
            print(f"  Copied adaptor layer: {key}")

        # layer_norm（使用group1的）
        elif 'layer_norm' in key and 'independent' not in key:
            merged_model['model'][key] = group1_model['model'][key]
            print(f"  Copied layer_norm: {key}")

    # 3. 复制group1的独立层权重
    for key in group1_model['model']:
        if 'independent_layers' in key and 'group1' in key:
            merged_model['model'][key] = group1_model['model'][key]
            print(f"  Copied group1 independent layer: {key}")

    # 4. 复制group2的独立层权重
    for key in group2_model['model']:
        if 'independent_layers' in key and 'group2' in key:
            merged_model['model'][key] = group2_model['model'][key]
            print(f"  Copied group2 independent layer: {key}")

    # 保存合并后的模型
    print(f"Saving merged model to {output_path}")
    torch.save(merged_model, output_path)

    # 打印统计信息
    print(f"Merged model statistics:")
    print(f"  Total keys: {len(merged_model['model'])}")

    # 统计各层的key数量
    frozen_keys = [k for k in merged_model['model'] if 'frozen_layers' in k]
    shared_keys = [k for k in merged_model['model'] if 'shared_training_layers' in k]
    group1_keys = [k for k in merged_model['model'] if 'independent_layers' in k and 'group1' in k]
    group2_keys = [k for k in merged_model['model'] if 'independent_layers' in k and 'group2' in k]

    print(f"  Frozen layer keys: {len(frozen_keys)}")
    print(f"  Shared training layer keys: {len(shared_keys)}")
    print(f"  Group1 independent layer keys: {len(group1_keys)}")
    print(f"  Group2 independent layer keys: {len(group2_keys)}")

    print("Model merged successfully!")


def parse_lang_groups(lang_groups_str):
    """解析语言分组字符串"""
    groups = {}
    for group_str in lang_groups_str.split(';'):
        group_name, langs = group_str.split(':')
        groups[group_name] = langs.split(',')
    return groups


def main():
    parser = argparse.ArgumentParser(description="Merge multi-language model weights")
    parser.add_argument("--group1_model", type=str, required=True,
                       help="Path to group1 model checkpoint")
    parser.add_argument("--group2_model", type=str, required=True,
                       help="Path to group2 model checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save merged model")
    parser.add_argument("--lang_groups", type=str, required=True,
                       help="Language groups in format 'group1:aeb,gle;group2:est,bem'")

    args = parser.parse_args()

    # 解析语言分组
    lang_groups = parse_lang_groups(args.lang_groups)
    print(f"Language groups: {lang_groups}")

    # 合并模型
    merge_models(
        group1_path=args.group1_model,
        group2_path=args.group2_model,
        output_path=args.output_path,
        lang_groups=lang_groups
    )


if __name__ == "__main__":
    main()
