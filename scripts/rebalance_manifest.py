
import json
import argparse
from collections import defaultdict

def load_manifest(file_path):
    """加载manifest文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    for line in lines:
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples

def group_by_language(samples):
    """按语言分组样本"""
    lang_groups = defaultdict(list)
    for sample in samples:
        lang = sample['source']['lang']
        lang_groups[lang].append(sample)
    return lang_groups

def distribute_samples(lang_groups):
    """均匀分布样本

    策略：
    1. 计算每种语言的间隔比例
    2. 使用轮询方式从各语言中选取样本
    3. 对于数量较少的语言，适当增加间隔
    """
    # 统计各语言数量
    lang_counts = {lang: len(samples) for lang, samples in lang_groups.items()}

    # 找出数量最多的语言作为基准
    max_count = max(lang_counts.values())

    # 计算每种语言应该出现的频率（相对于最大数量）
    # 例如：如果max_count=100，某语言有50个样本，则该语言每2个位置出现一次
    lang_intervals = {}
    for lang, count in lang_counts.items():
        interval = max_count / count
        lang_intervals[lang] = interval

    # 使用指针跟踪各语言的当前样本索引
    lang_pointers = {lang: 0 for lang in lang_groups}

    # 使用计数器跟踪各语言的累积间隔
    lang_counters = {lang: 0.0 for lang in lang_groups}

    distributed_samples = []
    total_samples = sum(lang_counts.values())

    while len(distributed_samples) < total_samples:
        # 找出计数器最小的语言
        min_lang = min(lang_counters, key=lang_counters.get)

        # 从该语言中取出一个样本
        lang = min_lang
        pointer = lang_pointers[lang]
        if pointer < len(lang_groups[lang]):
            distributed_samples.append(lang_groups[lang][pointer])
            lang_pointers[lang] += 1
            lang_counters[lang] += lang_intervals[lang]
        else:
            # 该语言的样本已经用完
            del lang_counters[lang]
            del lang_pointers[lang]

    return distributed_samples

def save_manifest(samples, output_path):
    """保存manifest文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='重新平衡manifest文件中的语言样本分布')
    parser.add_argument('--input', '-i', required=True, help='输入的manifest文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出的manifest文件路径')
    args = parser.parse_args()

    # 加载数据
    print(f"加载manifest文件: {args.input}")
    samples = load_manifest(args.input)
    print(f"总样本数: {len(samples)}")

    # 按语言分组
    lang_groups = group_by_language(samples)
    print("\n各语言样本数:")
    for lang, group in lang_groups.items():
        print(f"  {lang}: {len(group)}")

    # 重新分布
    print("\n重新分布样本...")
    distributed_samples = distribute_samples(lang_groups)

    # 保存结果
    print(f"\n保存到: {args.output}")
    save_manifest(distributed_samples, args.output)
    print("完成!")

if __name__ == '__main__':
    main()
