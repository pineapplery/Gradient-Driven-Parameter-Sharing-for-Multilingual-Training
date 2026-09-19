#!/usr/bin/env python3
"""
Gradient conflict analysis for B1 architecture with multiple language pairs.

Key modifications for B1:
1. Applies B1 architecture before loading checkpoint
2. Supports B1-specific parameters (r_shared, r_group)
3. Maintains all original gradient analysis features

- Reads per-language manifest JSONL files (manifest_dir/{split}_{lang_pair}_manifest.json).
- Loads model checkpoint with B1 architecture applied.
- For each language, iterates samples, forms single-language batches, computes loss (S2TT CE), accumulates gradients with `update_freq` and extracts a gradient vector per accumulation step.
- Saves per-language gradient vectors (.npy), computes cosine similarities between languages, outputs histograms and heatmaps, and writes summary JSON.

Usage example:
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
python /224040284/workspace/seamless_communication/src/seamless_communication/scripts/gradient_conflict_analysis_b1.py \
  --manifest_dir /224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests \
  --lang_pairs aeb_eng,bem_eng,est_eng,gle_eng \
  --split train \
  --checkpoint /224040284/workspace/seamless_communication/src/seamless_communication/output/checkpoint/b1_unify2/b1_checkpoint.pt \
  --output_dir /224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/b1_analysis \
  --batch_size 4 --update_freq 4 --max_samples 7000 --proj_dim 256 --proj_only_last_n_layers 2 \
  --param_level_norm --layer_level_norm --per_layer_analysis --freeze_encoder_except_last_n 2 \
  --num_workers 8 \
  --r_shared 2048 --r_group 1024

Notes:
- Expects manifest JSONL where each line has `source.audio_local_path` (pointing to .npy fbank) and `target.text`.
- B1 architecture is applied before loading checkpoint.
"""

import argparse
import csv
import itertools
import json
import logging
import math
import hashlib
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add B1 module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "seamless_m4t_medium"))
from b1_minimal import apply_b1_to_model, set_current_group, get_group_for_lang

# Use the same unity model and dataloader used in training to ensure feature dims match.
from seamless_communication.models.unity.loader import load_unity_model, load_unity_text_tokenizer, load_unity_unit_tokenizer
from seamless_communication.scripts.dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig
from fairseq2.nn.padding import PaddingMask
from fairseq2.models.sequence import SequenceModelOutput


def load_gradients_from_npy(npy_dir: Path, lang_pairs: List[str]) -> Dict[str, np.ndarray]:
    """
    Load pre-extracted gradient vectors from npy files.
    Expected file format: {lang_pair}_gradients.npy
    """
    lang_vectors = {}
    for lang_pair in lang_pairs:
        npy_file = npy_dir / f'{lang_pair}_gradients.npy'
        if npy_file.exists():
            arr = np.load(npy_file)
            lang_vectors[lang_pair] = arr
            print(f'✓ Loaded {lang_pair}: {npy_file}')
            print(f'  Shape: {arr.shape}, Size: {arr.nbytes / 1024 / 1024:.2f} MB')
        else:
            print(f'⚠️  Not found: {npy_file}')
    return lang_vectors


def load_manifest(manifest_path: Path, max_samples: int = None) -> List[Dict]:
    """Load manifest from JSONL file."""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def tokenize_texts(processor, texts: List[str], device: torch.device, max_length: int = 200):
    # processor assumed to have a tokenizer with huggingface API
    tok = processor.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tok['input_ids']
    if hasattr(processor.tokenizer, 'pad_token_id') and processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    return labels.to(device)


def extract_grad_blocks(model: torch.nn.Module, device: torch.device) -> Dict[str, 'torch.Tensor']:
    """
    Return a dict mapping parameter full-name -> flattened torch gradient (float32) on `device`.
    Only include parameters where p.requires_grad is True.
    """
    blocks: Dict[str, 'torch.Tensor'] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            t = torch.zeros(p.numel(), dtype=torch.float32, device=device)
        else:
            # ensure float32 tensor on requested device
            t = g.detach().to(device=device, dtype=torch.float32).reshape(-1)
        blocks[name] = t
    return blocks


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    # pooled standard deviation
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def _stable_seed_from_name(name: str) -> int:
    # deterministic 32-bit int seed from name
    h = hashlib.md5(name.encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'little')


def compute_and_save(sim_out_dir: Path, lang_vectors: Dict[str, np.ndarray], bins=200):
    # Create output directory at the beginning
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    
    norms = {k: l2_normalize_rows(v) for k, v in lang_vectors.items()}
    
    # Debug: Check normalized vectors
    import logging
    logger = logging.getLogger(__name__)
    for lang, norm_arr in norms.items():
        logger.info(f'After normalization {lang}:')
        logger.info(f'  Shape: {norm_arr.shape}')
        logger.info(f'  Mean: {norm_arr.mean():.6f}, Std: {norm_arr.std():.6f}')
        logger.info(f'  NaN count: {np.isnan(norm_arr).sum()}')
        logger.info(f'  Inf count: {np.isinf(norm_arr).sum()}')
    
    results_summary = {}

    for main_lang, V_main in norms.items():
        # 使用与gradient_conflict_analysis.py一致的文件名格式
        out_png = sim_out_dir / f"sim_hist_{main_lang}.png"
        plt.figure(figsize=(8, 6))
        for other_lang, V_other in norms.items():
            if V_main.size == 0 or V_other.size == 0:
                cos_vals = np.array([])
            else:
                cos_vals = (V_main @ V_other.T).ravel()
            counts, bin_edges = np.histogram(cos_vals, bins=bins, range=(-1, 1))
            center = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            plt.plot(center, counts, label=f"{main_lang} vs {other_lang}")
            results_summary.setdefault(main_lang, {})[other_lang] = {
                'mean': float(np.mean(cos_vals)) if cos_vals.size else None,
                'std': float(np.std(cos_vals)) if cos_vals.size else None,
                'count': int(cos_vals.size)
            }

        plt.title(f"Gradient cosine histogram ({main_lang} centered)")
        plt.xlabel('Cosine similarity')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    # Compute pairwise language similarity matrix (mean cosine) and save heatmap + CSV/JSON
    langs = list(norms.keys())
    n_lang = len(langs)
    mean_mat = np.zeros((n_lang, n_lang), dtype=np.float16)
    std_mat = np.zeros((n_lang, n_lang), dtype=np.float16)
    count_mat = np.zeros((n_lang, n_lang), dtype=np.int64)

    for i, li in enumerate(langs):
        for j, lj in enumerate(langs):
            # ⭐ FIXED: Compute ACTUAL self-similarity (not hardcoded 1.0)
            # This represents S_self in Method B (same language, different samples)
            if i == j:
                # Self similarity: same language, different samples (excluding diagonal)
                n_samples = norms[li].shape[0]
                if n_samples > 1:
                    # Compute full similarity matrix
                    sim_mat = norms[li] @ norms[li].T
                    # Extract off-diagonal elements (different samples from same language)
                    cos_vals = sim_mat[np.triu_indices_from(sim_mat, k=1)]
                    if len(cos_vals) > 0:
                        mean_mat[i, j] = cos_vals.mean()
                        std_mat[i, j] = cos_vals.std()
                    else:
                        mean_mat[i, j] = 0.0
                        std_mat[i, j] = 0.0
                else:
                    # Only one sample in language
                    mean_mat[i, j] = 0.0
                    std_mat[i, j] = 0.0
                count_mat[i, j] = n_samples
            else:
                # Cross similarity: different languages (represents S_cross in Method B)
                cos_vals = (norms[li] @ norms[lj].T).flatten()
                mean_mat[i, j] = cos_vals.mean()
                std_mat[i, j] = cos_vals.std()
                count_mat[i, j] = len(cos_vals)

    # CSVs already prepared to save (directory created at function start)
    csv_mean = sim_out_dir / 'pairwise_mean.csv'
    csv_std = sim_out_dir / 'pairwise_std.csv'
    csv_count = sim_out_dir / 'pairwise_count.csv'

    with open(csv_mean, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + langs)
        for i, li in enumerate(langs):
            writer.writerow([li] + [f"{mean_mat[i,j]:.6f}" if not np.isnan(mean_mat[i,j]) else '' for j in range(n_lang)])

    with open(csv_std, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + langs)
        for i, li in enumerate(langs):
            writer.writerow([li] + [f"{std_mat[i,j]:.6f}" if not np.isnan(std_mat[i,j]) else '' for j in range(n_lang)])

    with open(csv_count, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + langs)
        for i, li in enumerate(langs):
            writer.writerow([li] + [str(count_mat[i, j]) for j in range(n_lang)])

    # Save JSON summary for pairwise
    pairwise_summary = { 'langs': langs, 'mean': mean_mat.tolist(), 'std': std_mat.tolist(), 'count': count_mat.tolist() }
    with open(sim_out_dir / 'pairwise_summary.json', 'w', encoding='utf-8') as f:
        json.dump(pairwise_summary, f, indent=2)

    # Plot 4x4 heatmap of mean similarities
    heat_png = sim_out_dir / 'pairwise_heat_mean.png'
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_mat, xticklabels=langs, yticklabels=langs, cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.3f')
    plt.title('Pairwise mean cosine similarity (Sample-level: Self vs Cross)')
    plt.tight_layout()
    plt.savefig(heat_png, dpi=200)
    plt.close()

    # ⭐ ADD METHOD B ANALYSIS: Compute conflict indices for each language
    method_b_results = {}
    logger.info("\n" + "="*80)
    logger.info("METHOD B: Gradient Conflict Analysis (Sample-Level)")
    logger.info("="*80)
    
    for i, li in enumerate(langs):
        self_sim = mean_mat[i, i]
        # Cross similarity: average with all other languages
        cross_sims = [mean_mat[i, j] for j in range(n_lang) if i != j]
        cross_sim = np.mean(cross_sims) if cross_sims else 0.0
        conflict_index = self_sim - cross_sim
        
        # Map to shared ratio using piecewise function (from Method B in paper)
        if conflict_index < 0.15:
            shared_ratio = 0.75
        elif conflict_index < 0.30:
            shared_ratio = 0.50
        else:
            shared_ratio = 0.25
        
        method_b_results[li] = {
            'self_similarity': float(self_sim),
            'cross_similarity': float(cross_sim),
            'conflict_index': float(conflict_index),
            'recommended_shared_ratio': shared_ratio,
            'explanation': f"C = {conflict_index:.4f}: {'High' if conflict_index >= 0.30 else 'Moderate' if conflict_index >= 0.15 else 'Low'} conflict"
        }
        
        logger.info(f"\n{li}:")
        logger.info(f"  S_self (same lang):     {self_sim:.4f}")
        logger.info(f"  S_cross (diff langs):   {cross_sim:.4f}")
        logger.info(f"  Conflict Index C:       {conflict_index:.4f}")
        logger.info(f"  → Recommended ratio:    {shared_ratio*100:.0f}% shared / {(1-shared_ratio)*100:.0f}% private")
    
    # Save Method B analysis
    with open(sim_out_dir / 'method_b_conflict_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(method_b_results, f, indent=2)

    with open(sim_out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)


def run_global_embeddings_visuals(sim_out_dir: Path, lang_vectors: Dict[str, np.ndarray]):
    # Create output directory at the beginning
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    
    # PCA + t-SNE on the projected vectors (concatenate langs)
    all_vecs = []
    labels = []
    for lang, arr in lang_vectors.items():
        all_vecs.append(arr)
        labels.extend([lang] * arr.shape[0])
    if not all_vecs:
        return
    X = np.concatenate(all_vecs, axis=0)
    pca = PCA(n_components=min(10, X.shape[1]))
    Xp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    Xts = tsne.fit_transform(Xp[:, :min(10, Xp.shape[1])])

    plt.figure(figsize=(8, 6))
    uniq = sorted(set(labels))
    for u in uniq:
        mask = np.array(labels) == u
        plt.scatter(Xts[mask, 0], Xts[mask, 1], label=u, alpha=0.6, s=10)
    plt.legend()
    plt.title('t-SNE of projected gradient sketches')
    plt.tight_layout()
    plt.savefig(sim_out_dir / 'tsne_proj.png', dpi=200)
    plt.close()

    # PCA first two components
    plt.figure(figsize=(8, 6))
    Xp2 = pca.transform(X)[:, :2]
    for u in uniq:
        mask = np.array(labels) == u
        plt.scatter(Xp2[mask, 0], Xp2[mask, 1], label=u, alpha=0.6, s=10)
    plt.legend()
    plt.title('PCA (2D) of projected gradient sketches')
    plt.tight_layout()
    plt.savefig(sim_out_dir / 'pca_proj.png', dpi=200)
    plt.close()


def compute_and_save_layerwise(sim_out_dir: Path, lang_layer_vectors: Dict[str, Dict[int, np.ndarray]]):
    """
    lang_layer_vectors: { lang: { layer_idx: (N, proj_dim) ndarray } }
    Matches the original gradient_conflict_analysis.py structure exactly.
    """
    # Create output directory at the beginning
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    
    # collect all layer indices
    layer_set = set()
    langs = sorted(lang_layer_vectors.keys())
    for lang, mapping in lang_layer_vectors.items():
        for li in mapping.keys():
            layer_set.add(int(li))
    if not layer_set:
        return
    layers = sorted(layer_set)

    # Prepare results structure - use nested dict like original version
    layer_pairwise = {}
    dist_tests = {}

    for li in layers:
        layer_pairwise[li] = {}
        dist_tests[li] = {}
        
        # For each lang pair compute sims
        for i, la in enumerate(langs):
            A = lang_layer_vectors.get(la, {}).get(li)
            if A is None or A.size == 0:
                continue
            A_norm = l2_normalize_rows(A.astype(np.float32))
            for j, lb in enumerate(langs):
                B = lang_layer_vectors.get(lb, {}).get(li)
                if B is None or B.size == 0:
                    continue
                B_norm = l2_normalize_rows(B.astype(np.float32))
                sims = (A_norm @ B_norm.T).ravel()
                layer_pairwise[li].setdefault(la, {})[lb] = {
                    'mean': float(np.mean(sims)),
                    'std': float(np.std(sims)),
                    'count': int(sims.size)
                }

        # Distribution tests (within vs between) for each lang pair
        for la in langs:
            A = lang_layer_vectors.get(la, {}).get(li)
            if A is None or A.size == 0:
                continue
            A_norm = l2_normalize_rows(A.astype(np.float32))
            within = (A_norm @ A_norm.T).ravel()
            for lb in langs:
                if la == lb:
                    continue
                B = lang_layer_vectors.get(lb, {}).get(li)
                if B is None or B.size == 0:
                    continue
                B_norm = l2_normalize_rows(B.astype(np.float32))
                between = (A_norm @ B_norm.T).ravel()
                
                # KS test and t-test comparing within vs between
                try:
                    ks_stat, ks_p = stats.ks_2samp(within, between)
                except Exception:
                    ks_stat, ks_p = float('nan'), float('nan')
                try:
                    t_stat, t_p = stats.ttest_ind(within, between, equal_var=False)
                except Exception:
                    t_stat, t_p = float('nan'), float('nan')
                d = cohen_d(within, between)
                dist_tests[li].setdefault(f"{la}__vs__{lb}", {})["ks"] = {'stat': float(ks_stat), 'p': float(ks_p)}
                dist_tests[li].setdefault(f"{la}__vs__{lb}", {})["ttest"] = {'stat': float(t_stat), 'p': float(t_p)}
                dist_tests[li].setdefault(f"{la}__vs__{lb}", {})["cohen_d"] = float(d)

                # KDE + boxplot for the pair
                plt.figure(figsize=(8, 4))
                sns.kdeplot(within, label=f"{la} within", bw_method='scott')
                sns.kdeplot(between, label=f"{la}-{lb} between", bw_method='scott')
                plt.title(f"Layer {li} KDE: {la} within vs {la}-{lb} between")
                plt.legend()
                plt.tight_layout()
                plt.savefig(sim_out_dir / f"layer_{li}_kde_{la}_vs_{lb}.png", dpi=200)
                plt.close()

                plt.figure(figsize=(6, 4))
                sns.boxplot(data=[within, between])
                plt.xticks([0, 1], [f"{la} within", f"{la}-{lb} between"], rotation=45)
                plt.title(f"Layer {li} box: {la} within vs {la}-{lb} between")
                plt.tight_layout()
                plt.savefig(sim_out_dir / f"layer_{li}_box_{la}_vs_{lb}.png", dpi=200)
                plt.close()

    # Save JSON summaries
    with open(sim_out_dir / 'layer_pairwise_summary.json', 'w', encoding='utf-8') as f:
        json.dump(layer_pairwise, f, indent=2)
    with open(sim_out_dir / 'layer_distribution_tests.json', 'w', encoding='utf-8') as f:
        json.dump(dist_tests, f, indent=2)

    # Grouped bar plot across layers for cross-language pairs
    cross_pairs = list(itertools.combinations(langs, 2))
    # prepare matrix: rows=layers, cols=len(cross_pairs)
    data_mat = np.zeros((len(layers), len(cross_pairs)), dtype=np.float32)
    for li_idx, li in enumerate(layers):
        for pj, (la, lb) in enumerate(cross_pairs):
            val = None
            try:
                val = layer_pairwise[li][la][lb]['mean']
            except Exception:
                val = float('nan')
            data_mat[li_idx, pj] = val

    # plot grouped bars with zero baseline
    x = np.arange(len(layers))
    width = 0.8 / max(1, len(cross_pairs))
    plt.figure(figsize=(max(8, len(layers) * 0.6), 6))
    for pj, (la, lb) in enumerate(cross_pairs):
        plt.bar(x + pj * width, data_mat[:, pj], width=width, label=f"{la}-{lb}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xticks(x + width * (len(cross_pairs) - 1) / 2.0, [str(li) for li in layers])
    plt.xlabel('Layer index')
    plt.ylabel('Mean cosine similarity')
    plt.title('Per-layer cross-language mean cosine similarities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(sim_out_dir / 'per_layer_cross_lang_bars.png', dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='B1 Gradient Conflict Analysis')
    parser.add_argument('--manifest_dir', type=Path, required=True, help='Directory containing manifest files')
    parser.add_argument('--lang_pairs', type=str, required=True, help='Comma-separated language pairs (e.g., aeb_eng,bem_eng)')
    parser.add_argument('--split', type=str, default='train', help='Data split name')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to B1 checkpoint')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--update_freq', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per language')
    parser.add_argument('--proj_dim', type=int, default=256, help='Projection dimension')
    parser.add_argument('--proj_only_last_n_layers', type=int, default=None, help='Project only last N layers')
    parser.add_argument('--param_level_norm', action='store_true', help='Normalize at parameter level')
    parser.add_argument('--layer_level_norm', action='store_true', help='Normalize at layer level')
    parser.add_argument('--per_layer_analysis', action='store_true', help='Perform per-layer analysis')
    parser.add_argument('--freeze_encoder_except_last_n', type=int, default=None, help='Freeze encoder except last N layers')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--r_shared', type=int, default=2048, help='B1 shared FFN rank')
    parser.add_argument('--r_group', type=int, default=1024, help='B1 group FFN rank')
    parser.add_argument('--model_name', type=str, default='seamlessM4T_medium', help='Base model name')
    parser.add_argument('--load_from_npy', type=Path, default=None, help='Directory containing npy files (skip gradient extraction)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Parse language pairs
    lang_pairs = [lp.strip() for lp in args.lang_pairs.split(',')]
    logger.info(f'Language pairs: {lang_pairs}')
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============ 直接从npy文件加载梯度（跳过梯度提取） ============
    if args.load_from_npy is not None:
        logger.info('=' * 80)
        logger.info('FAST MODE: Loading pre-computed gradients from npy files')
        logger.info('=' * 80)
        
        if not args.load_from_npy.exists():
            logger.error(f'NPY directory not found: {args.load_from_npy}')
            return
        
        lang_vectors = load_gradients_from_npy(args.load_from_npy, lang_pairs)
        
        if not lang_vectors:
            logger.error('No gradient vectors loaded!')
            return
        
        logger.info(f'✓ Successfully loaded {len(lang_vectors)}/{len(lang_pairs)} language pairs')
        
        # Jump directly to similarity analysis
        logger.info('\n' + '='*80)
        logger.info('Computing Similarity Analysis')
        logger.info('='*80)
        sim_dir = args.output_dir / 'similarity_analysis'
        compute_and_save(sim_dir, lang_vectors)
        logger.info('✓ Similarity analysis complete')
        
        logger.info('\nRunning global embeddings visualization (PCA & t-SNE)...')
        run_global_embeddings_visuals(sim_dir, lang_vectors)
        logger.info('✓ Visualization complete')
        
        logger.info('\n' + '='*80)
        logger.info('✓ Gradient Analysis Complete!')
        logger.info(f'  Results saved to: {args.output_dir}')
        logger.info(f'  Languages analyzed: {len(lang_vectors)}/{len(lang_pairs)}')
        logger.info(f'  Total gradient vectors: {sum(v.shape[0] for v in lang_vectors.values())}')
        logger.info('='*80)
        return
    
    # ============ 正常模式：提取梯度 ============
    logger.info('=' * 80)
    logger.info('NORMAL MODE: Extracting gradients from data')
    logger.info('=' * 80)
    
    # Detailed GPU diagnosis
    logger.info('=' * 80)
    logger.info('GPU 诊断信息:')
    logger.info('=' * 80)
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'CUDA is_available: {torch.cuda.is_available()}')
    logger.info(f'CUDA device_count: {torch.cuda.device_count()}')
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f'  Device {i}: {torch.cuda.get_device_name(i)}')
    else:
        logger.warning('CUDA 不可用！请检查以下内容:')
        logger.warning('  1. 运行 nvidia-smi 检查GPU是否被识别')
        logger.warning('  2. 检查CUDA_VISIBLE_DEVICES环境变量设置是否正确')
        logger.warning('  3. 确保PyTorch版本支持当前的CUDA版本')
        logger.warning('  4. 尝试重新安装支持GPU的PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
        logger.warning('')
        logger.warning('继续使用CPU运行（会很慢，每个语言约30-60分钟）...')
    
    logger.info('=' * 80)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'✓ 使用设备: {device}')
    if device.type == 'cpu':
        logger.warning('⚠️  将在CPU上运行，7000个样本可能需要1-2小时/语言')
    
    # Parse language pairs
    lang_pairs = [lp.strip() for lp in args.lang_pairs.split(',')]
    logger.info(f'Language pairs: {lang_pairs}')
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizers
    logger.info('Loading tokenizers...')
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    # Load base model
    logger.info(f'Loading base model: {args.model_name}')
    model = load_unity_model(args.model_name, device=device, dtype=torch.float32)
    
    # Remove unnecessary modules
    if model.t2u_model is not None:
        logger.info('Removing t2u_model')
        model.t2u_model = None
    if model.text_encoder is not None:
        logger.info('Removing text_encoder')
        model.text_encoder = None
    
    # Apply B1 architecture
    logger.info(f'Applying B1 architecture (r_shared={args.r_shared}, r_group={args.r_group})')
    model = apply_b1_to_model(model, r_shared=args.r_shared, r_group=args.r_group)
    
    # Load B1 checkpoint
    logger.info(f'Loading B1 checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Strip prefix if present
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info(f'Checkpoint loaded. Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}')
    
    # Freeze parameters if specified
    if args.freeze_encoder_except_last_n is not None:
        logger.info(f'Freezing encoder except last {args.freeze_encoder_except_last_n} layers')
        # Get encoder layers
        if hasattr(model, 'speech_encoder'):
            speech_encoder = model.speech_encoder
        elif hasattr(model, 'model') and hasattr(model.model, 'speech_encoder'):
            speech_encoder = model.model.speech_encoder
        else:
            raise ValueError('Cannot find speech_encoder')
        
        if hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
            layers = speech_encoder.inner.layers
        elif hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
            layers = speech_encoder.encoder.layers
        elif hasattr(speech_encoder, 'layers'):
            layers = speech_encoder.layers
        else:
            raise ValueError('Cannot find encoder layers')
        
        total_layers = len(layers)
        freeze_until = total_layers - args.freeze_encoder_except_last_n
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers
        for i in range(freeze_until, total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)')
    
    # Move model to device
    model = model.to(device)
    model.train()  # Set to train mode for gradient computation
    
    # Process each language pair
    lang_vectors = {}
    lang_layer_vectors = {} if args.per_layer_analysis else None
    
    for lang_pair in lang_pairs:
        logger.info(f'\n=== Processing {lang_pair} ===')
        
        # Load manifest
        manifest_file = args.manifest_dir / f'{args.split}_{lang_pair}_manifest.json'
        if not manifest_file.exists():
            logger.warning(f'Manifest not found: {manifest_file}, skipping')
            continue
        
        samples = load_manifest(manifest_file, args.max_samples)
        logger.info(f'Loaded {len(samples)} samples')
        
        # Create dataloader
        batching_config = BatchingConfig(
            batch_size=args.batch_size,
            rank=0,
            world_size=1,
            max_audio_length_sec=15.0,
            float_dtype=torch.float32,
            use_fbank=True,
        )
        
        # Save temp manifest for dataloader
        temp_manifest = args.output_dir / f'temp_{lang_pair}_manifest.json'
        with open(temp_manifest, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        dataloader = UnitYDataLoaderWithFbank(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            batching_config=batching_config,
            dataset_manifest_path=str(temp_manifest),
        )
        
        # Extract gradients
        gradient_vectors = []
        layer_gradients = {li: [] for li in range(12)} if args.per_layer_analysis else None
        
        # Set language group for B1
        src_lang = lang_pair.split('_')[0]
        group = get_group_for_lang(src_lang)
        set_current_group(group)
        logger.info(f'Using B1 group: {group}')
        
        model.zero_grad()
        accum_count = 0
        n_samples = 0
        batch_count = 0
        
        # ⭐ 移到外层：投影矩阵缓存（跨所有batch重用）
        proj_cache: Dict[str, torch.Tensor] = {}
        
        # Track progress by samples, not batches
        pbar = tqdm(total=min(len(samples), args.max_samples or len(samples)), 
                   desc=f'{lang_pair}', unit='samples')
        
        for batch_idx, batch in enumerate(dataloader):
            # Check max samples limit
            if args.max_samples and n_samples >= args.max_samples:
                logger.info(f'Reached max_samples limit ({args.max_samples})')
                break
            
            try:
                # Extract tensors from S2T batch
                s2t_batch = batch.speech_to_text
                
                # Skip empty batches
                if s2t_batch.src_tokens is None:
                    continue
                    
                batch_size = s2t_batch.src_tokens.size(0)
                
                seqs = s2t_batch.src_tokens.to(device)
                seq_lens = s2t_batch.src_lengths.to(device)
                prev_output_tokens = s2t_batch.prev_output_tokens.to(device)
                target_seqs = s2t_batch.target_tokens.to(device)
                target_seq_lens = s2t_batch.target_lengths.to(device)
                
                # Forward pass
                with torch.enable_grad():
                    speech_encoder_out, speech_encoder_padding_mask = model.encode_speech(
                        seqs=seqs,
                        padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                    )
                    
                    text_decoder_out, text_decoder_padding_mask = model.decode(
                        seqs=prev_output_tokens,
                        padding_mask=PaddingMask(target_seq_lens, prev_output_tokens.size(1)),
                        encoder_output=speech_encoder_out,
                        encoder_padding_mask=speech_encoder_padding_mask,
                    )
                    
                    # Project to logits
                    assert model.final_proj is not None
                    text_logits = model.final_proj(text_decoder_out)
                    
                    # Compute loss
                    s2t_numel = torch.sum(target_seq_lens - 1).to(text_logits.device)
                    s2t_loss = SequenceModelOutput(logits=text_logits, vocab_info=model.target_vocab_info).compute_loss(
                        targets=target_seqs,
                        ignore_prefix_size=1,
                        label_smoothing=0.0,
                    )
                    loss = s2t_loss / s2t_numel
                    
                    # Backward with gradient accumulation scaling
                    loss = loss / args.update_freq
                    loss.backward()
                
                # Update sample count and progress
                n_samples += batch_size
                batch_count += 1
                accum_count += 1
                pbar.update(batch_size)
                pbar.set_postfix({
                    'loss': f'{loss.item()*args.update_freq:.4f}',
                    'accum': f'{accum_count}/{args.update_freq}',
                    'grads': len(gradient_vectors)
                })
                
                # Extract gradient after accumulation
                if accum_count >= args.update_freq:
                    # Extract gradient blocks
                    grad_blocks = extract_grad_blocks(model, device)
                    
                    if not grad_blocks:
                        logger.warning(f'No gradients extracted at batch {batch_count}')
                        model.zero_grad()
                        accum_count = 0
                        continue
                    
                    # ⭐ Filter last N layers if specified (MUST match gradient_analysis_advanced_b1.py)
                    if args.proj_only_last_n_layers is not None:
                        filtered_blocks = {}
                        for name, grad in grad_blocks.items():
                            for li in range(12 - args.proj_only_last_n_layers, 12):
                                if f'layers.{li}.' in name or f'layers[{li}]' in name:
                                    filtered_blocks[name] = grad
                                    break
                        grad_blocks = filtered_blocks
                        if not grad_blocks:
                            logger.warning(f'No gradients in last {args.proj_only_last_n_layers} layers')
                            model.zero_grad()
                            accum_count = 0
                            continue
                    
                    # Log gradient block info (first time only)
                    if len(gradient_vectors) == 0:
                        total_params = sum(v.numel() for v in grad_blocks.values())
                        logger.info(f'Gradient extraction: {len(grad_blocks)} parameter blocks, {total_params:,} total params')
                        logger.info(f'Projection: {total_params:,} -> {args.proj_dim}')
                    
                    # ⭐ Per-parameter projection + caching (已在外层创建proj_cache，此处直接使用)
                    proj_acc = torch.zeros(args.proj_dim, dtype=torch.float32, device=device)
                    per_layer_acc: Dict[int, torch.Tensor] = {}
                    
                    for param_name, grad in grad_blocks.items():
                        if grad.numel() == 0:
                            continue
                        
                        # Optional: param-level normalization
                        if args.param_level_norm:
                            grad_norm = grad.norm()
                            if grad_norm.item() > 0:
                                grad = grad / grad_norm
                        
                        # 获取或创建该参数的投影矩阵（缓存）
                        if param_name not in proj_cache:
                            seed = _stable_seed_from_name(param_name)
                            # 使用与原版完全相同的投影矩阵生成方式：高斯分布，不归一化
                            try:
                                rgen = np.random.RandomState(seed)
                                proj_matrix_np = rgen.normal(loc=0.0, scale=1.0, size=(args.proj_dim, int(grad.numel()))).astype(np.float32)
                            except Exception:
                                # skip extremely large blocks defensively
                                logger.warning(f'Failed to generate projection for {param_name}')
                                continue
                            
                            proj_matrix = torch.from_numpy(proj_matrix_np).to(
                                device=device, dtype=torch.float32
                            )
                            proj_cache[param_name] = proj_matrix
                        else:
                            proj_matrix = proj_cache[param_name]
                        
                        # 对单个参数梯度投影
                        try:
                            partial_proj = torch.matmul(proj_matrix, grad)
                            proj_acc += partial_proj
                        except RuntimeError:
                            # in rare cases of mismatched sizes, skip
                            continue
                        
                        # 提取层索引进行per-layer统计
                        m = re.search(r'layers\.(\d+)|layers\[(\d+)\]', param_name)
                        if m:
                            layer_idx = int(m.group(1) or m.group(2))
                            if layer_idx not in per_layer_acc:
                                per_layer_acc[layer_idx] = torch.zeros(
                                    args.proj_dim, dtype=torch.float32, device=device
                                )
                            per_layer_acc[layer_idx] += partial_proj
                    
                    # Normalize final projection
                    proj_norm = proj_acc.norm()
                    if proj_norm.item() > 0:
                        proj_acc = proj_acc / proj_norm
                    
                    gradient_vectors.append(proj_acc.cpu().detach().numpy())
                    
                    # Per-layer analysis (optional)
                    if args.per_layer_analysis:
                        for li in per_layer_acc:
                            ln = per_layer_acc[li].norm()
                            if ln.item() > 0:
                                per_layer_acc[li] = per_layer_acc[li] / ln
                            layer_gradients[li].append(per_layer_acc[li].cpu().detach().numpy())
                    
                    # Reset accumulation
                    model.zero_grad()
                    accum_count = 0
                    
                    # Log progress every 100 gradient vectors
                    if len(gradient_vectors) % 100 == 0:
                        logger.info(f'{lang_pair}: Collected {len(gradient_vectors)} gradient vectors from {n_samples} samples')
                
            except Exception as e:
                logger.error(f'Error processing batch {batch_idx}: {str(e)}')
                logger.error(traceback.format_exc())
                model.zero_grad()
                accum_count = 0
                continue
        
        pbar.close()
        
        logger.info(f'{lang_pair}: Processed {n_samples} samples in {batch_count} batches')
        logger.info(f'{lang_pair}: Collected {len(gradient_vectors)} gradient vectors')
        
        # Save gradient vectors
        if gradient_vectors:
            grad_array = np.array(gradient_vectors)
            lang_vectors[lang_pair] = grad_array
            save_path = args.output_dir / f'{lang_pair}_gradients.npy'
            np.save(save_path, grad_array)
            logger.info(f'✓ Saved gradient vectors: {save_path}')
            logger.info(f'  Shape: {grad_array.shape}, Size: {grad_array.nbytes / 1024 / 1024:.2f} MB')
        else:
            logger.warning(f'No gradient vectors collected for {lang_pair}!')
        
        # Save per-layer gradients
        if args.per_layer_analysis and layer_gradients:
            lang_layer_vectors[lang_pair] = {}
            layer_count = 0
            for li, grads in layer_gradients.items():
                if grads:
                    grad_array = np.array(grads)
                    lang_layer_vectors[lang_pair][li] = grad_array
                    np.save(args.output_dir / f'{lang_pair}_layer{li}_gradients.npy', grad_array)
                    layer_count += 1
            logger.info(f'✓ Saved per-layer gradients for {layer_count} layers')
        
        # Clean up temp manifest
        temp_manifest.unlink(missing_ok=True)
    
    # Compute and save similarity analysis
    if lang_vectors:
        logger.info('\n' + '='*80)
        logger.info('Computing Similarity Analysis')
        logger.info('='*80)
        
        # Debug: Check gradient vector statistics
        for lang, arr in lang_vectors.items():
            logger.info(f'{lang}:')
            logger.info(f'  Shape: {arr.shape}')
            logger.info(f'  Mean: {arr.mean():.6f}, Std: {arr.std():.6f}')
            logger.info(f'  Min: {arr.min():.6f}, Max: {arr.max():.6f}')
            logger.info(f'  NaN count: {np.isnan(arr).sum()}')
        
        sim_dir = args.output_dir / 'similarity_analysis'
        compute_and_save(sim_dir, lang_vectors)
        logger.info('✓ Similarity analysis complete')
        
        logger.info('\nRunning global embeddings visualization (PCA & t-SNE)...')
        run_global_embeddings_visuals(sim_dir, lang_vectors)
        logger.info('✓ Visualization complete')
    else:
        logger.warning('No gradient vectors collected - skipping similarity analysis')
    
    # Per-layer analysis
    if args.per_layer_analysis and lang_layer_vectors:
        logger.info('\n' + '='*80)
        logger.info('Computing Per-Layer Analysis')
        logger.info('='*80)
        compute_and_save_layerwise(sim_dir, lang_layer_vectors)
        logger.info('✓ Per-layer analysis complete')
    
    logger.info('\n' + '='*80)
    logger.info('✓ Gradient Conflict Analysis Complete!')
    logger.info(f'  Results saved to: {args.output_dir}')
    logger.info(f'  Languages analyzed: {len(lang_vectors)}/{len(lang_pairs)}')
    logger.info(f'  Total gradient vectors: {sum(v.shape[0] for v in lang_vectors.values())}')
    logger.info('='*80)


if __name__ == '__main__':
    main()
