#!/usr/bin/env python3
"""
多语言 S2T 翻译评估脚本
支持批量评估多个语言的翻译质量
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入评估工具
import sacrebleu
from sacrebleu.metrics import TER, CHRF
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint


def clean_text(text, lowercase=True, keep_punct=False):
    """
    清洗文本：可选转小写 + 可选保留标点符号
    
    Args:
        text: 输入文本
        lowercase: 是否转为小写 (默认: True)
        keep_punct: 是否保留标点符号 (默认: False)
    """
    text = str(text)
    
    if lowercase:
        text = text.lower()
    
    if not keep_punct:
        text = re.sub(r'[^\w\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean_data(ref_path, pred_path, lowercase=True, keep_punct=False):
    """
    加载并清洗参考文本和预测文本
    """
    print(f"  📂 加载参考文本: {ref_path}")
    df_ref = pd.read_csv(ref_path, sep='\t')
    
    print(f"  📂 加载预测文本: {pred_path}")
    df_pred = pd.read_csv(pred_path, sep='\t')
    
    print(f"  🧹 清洗参考文本... (lowercase={lowercase}, keep_punct={keep_punct})")
    df_ref['tgt_text'] = df_ref['tgt_text'].apply(
        lambda x: clean_text(x, lowercase=lowercase, keep_punct=keep_punct)
    )
    
    print(f"  🧹 清洗预测文本... (lowercase={lowercase}, keep_punct={keep_punct})")
    df_pred['text'] = df_pred['text'].apply(
        lambda x: clean_text(x, lowercase=lowercase, keep_punct=keep_punct)
    )
    
    # 合并数据
    print(f"  🔗 合并数据...")
    merged = pd.merge(df_ref, df_pred, left_on='audio', right_on='audio', how='inner')
    merged = merged.rename(columns={'tgt_text': 'reference', 'text': 'prediction'})
    
    # 处理缺失值
    merged = merged.dropna(subset=['reference', 'prediction'])
    
    print(f"  ✓ 成功匹配 {len(merged)} 个样本")
    return merged


def compute_bleu(predictions, references):
    """计算 BLEU 分数"""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_ter(predictions, references):
    """计算 TER 分数"""
    ter = TER().corpus_score(predictions, [references])
    return ter.score


def compute_chrf(predictions, references):
    """计算 CHRF 分数"""
    chrf = CHRF().corpus_score(predictions, [references])
    return chrf.score


def compute_chrfpp(predictions, references):
    """计算 CHRF++ 分数"""
    chrfpp = CHRF(word_order=2).corpus_score(predictions, [references])
    return chrfpp.score


def compute_bertscore(predictions, references):
    """计算 BERTScore"""
    P, R, F1 = bert_score(predictions, references, lang='en', rescale_with_baseline=True)
    return F1.mean().item()


def compute_comet(predictions, references, comet_model, batch_size=32):
    """计算 COMET 分数"""
    comet_scores = comet_model.predict(
        [
            {'src': '', 'mt': pred, 'ref': ref}
            for pred, ref in zip(predictions, references)
        ],
        batch_size=batch_size,
        gpus=1
    )
    return np.mean(comet_scores['scores'])


def evaluate_language(lang_code, ref_dir, pred_dir, comet_model, lowercase=True, keep_punct=False):
    """
    评估单个语言
    """
    print(f"\n{'='*60}")
    print(f"📊 评估语言: {lang_code.upper()}")
    print(f"{'='*60}")
    
    # 构建文件路径
    ref_path = os.path.join(ref_dir, f"test_{lang_code}_eng_npy.tsv")
    pred_path = os.path.join(pred_dir, f"output_{lang_code}_eng_npy_results.tsv")
    
    # 检查文件存在性
    if not os.path.exists(ref_path):
        print(f"  ❌ 参考文件不存在: {ref_path}")
        return None
    
    if not os.path.exists(pred_path):
        print(f"  ❌ 预测文件不存在: {pred_path}")
        return None
    
    # 加载和清洗数据
    try:
        merged = load_and_clean_data(ref_path, pred_path, lowercase=lowercase, keep_punct=keep_punct)
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return None
    
    # 提取预测和参考文本
    predictions = merged['prediction'].tolist()
    references = merged['reference'].tolist()
    
    # 计算评估指标
    print(f"\n  🔄 计算 BLEU...")
    bleu_score = compute_bleu(predictions, references)
    print(f"    ✓ BLEU: {bleu_score:.4f}")
    
    print(f"  🔄 计算 TER...")
    ter_score = compute_ter(predictions, references)
    print(f"    ✓ TER: {ter_score:.4f}")
    
    print(f"  🔄 计算 CHRF...")
    chrf_score = compute_chrf(predictions, references)
    print(f"    ✓ CHRF: {chrf_score:.4f}")
    
    print(f"  🔄 计算 CHRF++...")
    chrfpp_score = compute_chrfpp(predictions, references)
    print(f"    ✓ CHRF++: {chrfpp_score:.4f}")
    
    print(f"  🔄 计算 BERTScore...")
    bert_score_val = compute_bertscore(predictions, references)
    print(f"    ✓ BERTScore F1: {bert_score_val:.4f}")
    
    print(f"  🔄 计算 COMET...")
    comet_score = compute_comet(predictions, references, comet_model, batch_size=32)
    print(f"    ✓ COMET: {comet_score:.4f}")
    
    # 返回结果
    results = {
        'language': lang_code,
        'num_samples': len(merged),
        'metrics': {
            'BLEU': round(bleu_score, 4),
            'TER': round(ter_score, 4),
            'CHRF': round(chrf_score, 4),
            'CHRF++': round(chrfpp_score, 4),
            'BERTScore_F1': round(bert_score_val, 4),
            'COMET': round(comet_score, 4)
        }
    }
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多语言 S2T 翻译评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python evaluate_multilang_s2t.py \\
    --ref_dir /224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big \\
    --pred_dir /224040284/workspace/seamless_communication/src/seamless_communication/output/group_finetuned/B1_mix_2 \\
    --output_dir /224040284/workspace/seamless_communication/src/seamless_communication/output/group_finetuned/B1_mix_2
        """
    )
    
    parser.add_argument(
        '--ref_dir',
        type=str,
        default='/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big',
        help='参考翻译文件目录 (默认: data_big)'
    )
    
    parser.add_argument(
        '--pred_dir',
        type=str,
        required=True,
        help='预测结果文件目录 (必需参数)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出 JSON 结果文件的目录 (必需参数)'
    )
    
    parser.add_argument(
        '--languages',
        type=str,
        default='aeb,bem,est,gle',
        help='要评估的语言 (逗号分隔，默认: aeb,bem,est,gle)'
    )
    
    parser.add_argument(
        '--no_lowercase',
        action='store_true',
        default=False,
        help='不转为小写 (默认: 转为小写)'
    )
    
    parser.add_argument(
        '--keep_punct',
        action='store_true',
        default=False,
        help='保留标点符号 (默认: 删除标点)'
    )
    
    args = parser.parse_args()
    
    # 验证目录
    if not os.path.exists(args.ref_dir):
        print(f"❌ 参考文件目录不存在: {args.ref_dir}")
        return
    
    if not os.path.exists(args.pred_dir):
        print(f"❌ 预测文件目录不存在: {args.pred_dir}")
        return
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 解析语言列表
    languages = [lang.strip() for lang in args.languages.split(',')]
    
    print(f"\n{'='*60}")
    print(f"🚀 多语言 S2T 翻译评估")
    print(f"{'='*60}")
    print(f"📁 参考文件目录: {args.ref_dir}")
    print(f"📁 预测文件目录: {args.pred_dir}")
    print(f"📁 输出结果目录: {args.output_dir}")
    print(f"🌍 要评估的语言: {', '.join(languages)}")
    print(f"🔤 文本处理: lowercase={not args.no_lowercase}, keep_punct={args.keep_punct}")
    print(f"{'='*60}")
    
    # 下载 COMET 模型 (仅一次)
    print(f"\n⏳ 加载 COMET 模型...")
    try:
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_model_path)
        print(f"  ✓ COMET 模型加载成功")
    except Exception as e:
        print(f"  ⚠️ COMET 模型加载失败: {e}")
        print(f"  ⚠️ 将跳过 COMET 评分")
        comet_model = None
    
    # 评估每个语言
    all_results = {}
    failed_languages = []
    
    for lang in languages:
        try:
            result = evaluate_language(
                lang, args.ref_dir, args.pred_dir, comet_model,
                lowercase=not args.no_lowercase,
                keep_punct=args.keep_punct
            )
            if result:
                all_results[lang] = result
            else:
                failed_languages.append(lang)
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
            failed_languages.append(lang)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    print(f"\n{'='*60}")
    print(f"💾 保存结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 结果已保存")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"📈 评估汇总")
    print(f"{'='*60}")
    
    print(f"\n✓ 成功评估的语言:")
    for lang in all_results:
        metrics = all_results[lang]['metrics']
        print(f"\n  🌍 {lang.upper()} ({all_results[lang]['num_samples']} 样本)")
        print(f"    • BLEU:      {metrics['BLEU']:.4f}")
        print(f"    • TER:       {metrics['TER']:.4f}")
        print(f"    • CHRF:      {metrics['CHRF']:.4f}")
        print(f"    • CHRF++:    {metrics['CHRF++']:.4f}")
        print(f"    • BERTScore: {metrics['BERTScore_F1']:.4f}")
        print(f"    • COMET:     {metrics['COMET']:.4f}")
    
    if failed_languages:
        print(f"\n❌ 失败的语言: {', '.join(failed_languages)}")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
