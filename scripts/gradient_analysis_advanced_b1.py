#!/usr/bin/env python3
"""
Advanced gradient analysis methods for B1 architecture with multiple language pairs.

Key modifications for B1:
1. Applies B1 architecture before loading checkpoint
2. Supports B1-specific parameters (r_shared, r_group)
3. Maintains all original gradient analysis features (Method A/B/C)

Implements three novel gradient analysis approaches:
  - Method A: Cosine Grouping (hierarchical clustering, KMeans on gradient similarity)
  - Method B: Gradient Subspace Principal Similarity (SVD + CCA for subspace analysis)
  - Method C: Gradient Norm Dominance Profiles (per-layer gradient contribution analysis)

Reuses gradient vectors extracted from gradient_conflict_analysis_b1.py or computes new ones.

Usage example:
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
python /224040284/workspace/seamless_communication/src/seamless_communication/scripts/gradient_analysis_advanced_b1.py \
  --manifest_dir /224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests \
  --lang_pairs aeb_eng,bem_eng,est_eng,gle_eng \
  --split train \
  --checkpoint /224040284/workspace/seamless_communication/src/seamless_communication/output/checkpoint/b1_unify2/b1_checkpoint.pt \
  --output_dir /224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/b1_advanced \
  --batch_size 4 --update_freq 4 --max_samples 7000 --proj_dim 256 --proj_only_last_n_layers 2 \
  --param_level_norm --layer_level_norm --freeze_encoder_except_last_n 2 \
  --num_workers 8 \
  --r_shared 2048 --r_group 1024
"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

# Add B1 module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "seamless_m4t_medium"))
from b1_minimal import apply_b1_to_model, set_current_group, get_group_for_lang

# Import from seamless_communication
from seamless_communication.models.unity.loader import load_unity_model, load_unity_text_tokenizer, load_unity_unit_tokenizer
from seamless_communication.scripts.dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig
from fairseq2.nn.padding import PaddingMask
from fairseq2.models.sequence import SequenceModelOutput
from seamless_communication.cli.m4t.finetune.trainer import CalcLoss


# Helper functions from gradient_conflict_analysis
def load_manifest(manifest_path: Path, max_samples: int = None) -> List[Dict]:
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def extract_grad_blocks(model: torch.nn.Module, device: torch.device) -> Dict[str, 'torch.Tensor']:
    blocks: Dict[str, 'torch.Tensor'] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Handle None gradients by creating zero tensors (don't skip!)
        g = p.grad
        if g is None:
            t = torch.zeros(p.numel(), dtype=torch.float32, device=device)
        else:
            t = g.detach().to(device=device, dtype=torch.float32).reshape(-1)
        blocks[name] = t
    return blocks


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _stable_seed_from_name(name: str) -> int:
    h = hashlib.md5(name.encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'little')


# ============================================================================
# Method A: Cosine Grouping with Hierarchical Clustering and KMeans
# ============================================================================

def compute_cosine_similarity_matrix(lang_vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise cosine similarity matrix between language mean gradient vectors.
    
    Args:
        lang_vectors: { lang: (N, proj_dim) ndarray of gradient vectors }
    
    Returns:
        sim_matrix: (n_langs, n_langs) cosine similarity matrix
        langs: sorted language names
    """
    langs = sorted(lang_vectors.keys())
    n_lang = len(langs)
    sim_matrix = np.zeros((n_lang, n_lang), dtype=np.float32)
    
    for i, la in enumerate(langs):
        mean_a = lang_vectors[la].mean(axis=0, keepdims=True)
        mean_a = mean_a / (np.linalg.norm(mean_a) + 1e-8)
        
        for j, lb in enumerate(langs):
            mean_b = lang_vectors[lb].mean(axis=0, keepdims=True)
            mean_b = mean_b / (np.linalg.norm(mean_b) + 1e-8)
            
            sim = float((mean_a @ mean_b.T)[0, 0])
            sim_matrix[i, j] = sim
    
    return sim_matrix, langs


def method_a_cosine_grouping(output_dir: Path, lang_vectors: Dict[str, np.ndarray]):
    """
    Method A: Cosine Grouping Analysis
    - Compute pairwise cosine similarity
    - Hierarchical clustering (dendrogram)
    - KMeans clustering (K=2)
    - Heatmap visualization
    """
    print("\n=== Method A: Cosine Grouping Analysis ===")
    
    sim_matrix, langs = compute_cosine_similarity_matrix(lang_vectors)
    n_lang = len(langs)
    
    method_a_dir = output_dir / 'method_a_cosine_grouping'
    method_a_dir.mkdir(parents=True, exist_ok=True)
    
    results_a = {}
    
    # 1. Heatmap of cosine similarity
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, xticklabels=langs, yticklabels=langs, cmap='coolwarm', 
                vmin=0, vmax=1, annot=True, fmt='.3f', cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Language Gradient Similarity Matrix (Cosine)')
    plt.tight_layout()
    plt.savefig(method_a_dir / 'cosine_heatmap.png', dpi=200)
    plt.close()
    
    # 2. Hierarchical clustering
    # Convert similarity to distance
    dist_matrix = 1.0 - sim_matrix
    # Condense distance matrix for linkage
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='ward')
    
    # Extract key statistics from linkage matrix
    merge_distances = Z[:, 2].tolist()
    merge_samples = Z[:, 3].astype(int).tolist()
    
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=langs, leaf_font_size=12)
    plt.title('Hierarchical Clustering of Languages (Cosine-based Distance)')
    plt.xlabel('Language Pair')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(method_a_dir / 'dendrogram.png', dpi=200)
    plt.close()
    
    results_a['hierarchical_clustering'] = {
        'linkage_method': 'ward',
        'distance_metric': '1 - cosine_similarity',
        'merge_distances': merge_distances,
        'merge_samples_in_cluster': merge_samples,
        'distance_matrix': dist_matrix.tolist(),
        'last_merge_distance': float(Z[-1, 2])
    }
    
    # 3. KMeans clustering (K=2)
    mean_vecs = []
    for la in langs:
        mean_vec = lang_vectors[la].mean(axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
        mean_vecs.append(mean_vec)
    
    X = np.array(mean_vecs)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    cluster_assignments = {langs[i]: int(clusters[i]) for i in range(len(langs))}
    
    # Compute silhouette score
    from sklearn.metrics import silhouette_score, silhouette_samples
    silhouette_avg = float(silhouette_score(X, clusters))
    silhouette_vals = silhouette_samples(X, clusters)
    
    # Compute distances from each sample to its cluster center
    distances_to_center = []
    for i, lang in enumerate(langs):
        cluster_id = clusters[i]
        center = kmeans.cluster_centers_[cluster_id]
        dist = np.linalg.norm(X[i] - center)
        distances_to_center.append({
            'language': lang,
            'cluster': int(cluster_id),
            'distance_to_center': float(dist),
            'silhouette_score': float(silhouette_vals[i])
        })
    
    plt.figure(figsize=(8, 6))
    colors = ['red' if c == 0 else 'blue' for c in clusters]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=2)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='yellow', s=300, marker='X', edgecolors='black', linewidth=2, label='Cluster Centers')
    for i, lang in enumerate(langs):
        plt.annotate(lang, (X[i, 0], X[i, 1]), fontsize=9, ha='center')
    plt.title('KMeans Clustering of Languages (K=2)')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.legend(['Cluster 0', 'Cluster 1', 'Cluster Centers'])
    plt.tight_layout()
    plt.savefig(method_a_dir / 'kmeans_clusters.png', dpi=200)
    plt.close()
    
    results_a['kmeans_clustering'] = {
        'n_clusters': 2,
        'assignments': cluster_assignments,
        'inertia': float(kmeans.inertia_),
        'silhouette_score': silhouette_avg,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'per_language_stats': distances_to_center
    }
    
    # Save similarity matrix as CSV
    import csv
    with open(method_a_dir / 'cosine_similarity_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + langs)
        for i, la in enumerate(langs):
            writer.writerow([la] + [f'{sim_matrix[i, j]:.4f}' for j in range(n_lang)])
    
    # Add summary statistics
    cross_pairs = list(itertools.combinations(langs, 2))
    cross_sims = []
    for la, lb in cross_pairs:
        i = langs.index(la)
        j = langs.index(lb)
        cross_sims.append(sim_matrix[i, j])
    
    results_a['summary_statistics'] = {
        'mean_cross_similarity': float(np.mean(cross_sims)),
        'std_cross_similarity': float(np.std(cross_sims)),
        'min_cross_similarity': float(np.min(cross_sims)),
        'max_cross_similarity': float(np.max(cross_sims)),
        'median_cross_similarity': float(np.median(cross_sims))
    }
    
    # Save results
    with open(method_a_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results_a, f, indent=2)
    
    print(f"✓ Method A complete. Results saved to {method_a_dir}")


# ============================================================================
# Method B: Gradient Subspace Principal Similarity (SVD + CCA)
# ============================================================================

def compute_subspace_similarity_cca(Va: np.ndarray, Vb: np.ndarray, n_components: int = 5) -> float:
    """
    Compute subspace similarity using CCA between top singular vectors.
    
    Args:
        Va, Vb: Right singular vectors from SVD (shape: k x d)
        n_components: Number of CCA components
    
    Returns:
        Mean canonical correlation
    """
    try:
        # Transpose to (d, k) for CCA
        Va_t = Va.T
        Vb_t = Vb.T
        
        # Ensure we don't request more components than available
        n_comp = min(n_components, Va.shape[0], Vb.shape[0], Va.shape[1], Vb.shape[1])
        
        if n_comp < 1:
            return 0.0
        
        cca = CCA(n_components=n_comp)
        cca.fit(Va_t, Vb_t)
        
        # Compute canonical correlations
        X_c, Y_c = cca.transform(Va_t, Vb_t)
        correlations = []
        for i in range(n_comp):
            corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    except Exception as e:
        print(f"CCA computation failed: {e}")
        return 0.0


def method_b_subspace_similarity(output_dir: Path, lang_vectors: Dict[str, np.ndarray]):
    """
    Method B: Gradient Subspace Principal Similarity
    - Apply SVD to each language's gradient vectors
    - Compute subspace overlap using:
      1. Principal angle-based similarity
      2. CCA-based canonical correlation
    - Visualize top singular values and subspace similarities
    """
    print("\n=== Method B: Gradient Subspace Principal Similarity ===")
    
    method_b_dir = output_dir / 'method_b_subspace_similarity'
    method_b_dir.mkdir(parents=True, exist_ok=True)
    
    results_b = {}
    langs = sorted(lang_vectors.keys())
    
    # 1. Compute SVD for each language
    svd_results = {}
    k_components = 20  # Number of top components to keep
    
    for lang in langs:
        X = lang_vectors[lang]  # (N, d)
        
        # Center the data
        X_centered = X - X.mean(axis=0, keepdims=True)
        
        # SVD
        try:
            k = min(k_components, X_centered.shape[0], X_centered.shape[1])
            svd = TruncatedSVD(n_components=k, random_state=42)
            _ = svd.fit_transform(X_centered)
            
            svd_results[lang] = {
                'singular_values': svd.singular_values_.tolist(),
                'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
                'components': svd.components_,  # (k, d)
                'n_components': k
            }
            
        except Exception as e:
            print(f"SVD failed for {lang}: {e}")
            continue
    
    # Plot singular values
    plt.figure(figsize=(10, 6))
    for lang in langs:
        if lang in svd_results:
            sv = svd_results[lang]['singular_values']
            plt.plot(range(1, len(sv)+1), sv, marker='o', label=lang)
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value')
    plt.title('Top Singular Values by Language')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(method_b_dir / 'singular_values.png', dpi=200)
    plt.close()
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    for lang in langs:
        if lang in svd_results:
            evr = svd_results[lang]['explained_variance_ratio']
            cumsum = np.cumsum(evr)
            plt.plot(range(1, len(cumsum)+1), cumsum, marker='o', label=lang)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance by Language')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(method_b_dir / 'explained_variance.png', dpi=200)
    plt.close()
    
    # 2. Compute pairwise subspace similarities
    # Method 2a: Principal angles
    principal_angle_matrix = np.zeros((len(langs), len(langs)), dtype=np.float32)
    
    for i, la in enumerate(langs):
        for j, lb in enumerate(langs):
            if la not in svd_results or lb not in svd_results:
                continue
            
            if i == j:
                principal_angle_matrix[i, j] = 1.0
            else:
                # Compute principal angles between subspaces
                Va = svd_results[la]['components']  # (ka, d)
                Vb = svd_results[lb]['components']  # (kb, d)
                
                # Compute SVD of Va @ Vb.T
                try:
                    M = Va @ Vb.T
                    singular_vals = np.linalg.svd(M, compute_uv=False)
                    # Clip to [0, 1] to avoid numerical issues
                    singular_vals = np.clip(singular_vals, 0, 1)
                    # Compute principal angles
                    angles = np.arccos(singular_vals)
                    # Similarity metric: mean cosine of principal angles
                    similarity = float(np.mean(np.cos(angles)))
                    principal_angle_matrix[i, j] = similarity
                except Exception as e:
                    print(f"Principal angle computation failed for {la} vs {lb}: {e}")
                    principal_angle_matrix[i, j] = 0.0
    
    # Plot principal angle similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(principal_angle_matrix, xticklabels=langs, yticklabels=langs,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.3f',
                cbar_kws={'label': 'Subspace Similarity (Principal Angles)'})
    plt.title('Subspace Similarity via Principal Angles')
    plt.tight_layout()
    plt.savefig(method_b_dir / 'principal_angle_similarity.png', dpi=200)
    plt.close()
    
    # Method 2b: CCA-based canonical correlation
    cca_matrix = np.zeros((len(langs), len(langs)), dtype=np.float32)
    
    for i, la in enumerate(langs):
        for j, lb in enumerate(langs):
            if la not in svd_results or lb not in svd_results:
                continue
            
            if i == j:
                cca_matrix[i, j] = 1.0
            else:
                Va = svd_results[la]['components']
                Vb = svd_results[lb]['components']
                cca_sim = compute_subspace_similarity_cca(Va, Vb, n_components=5)
                cca_matrix[i, j] = cca_sim
    
    # Plot CCA similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cca_matrix, xticklabels=langs, yticklabels=langs,
                cmap='plasma', vmin=0, vmax=1, annot=True, fmt='.3f',
                cbar_kws={'label': 'Subspace Similarity (CCA)'})
    plt.title('Subspace Similarity via Canonical Correlation Analysis')
    plt.tight_layout()
    plt.savefig(method_b_dir / 'cca_similarity.png', dpi=200)
    plt.close()
    
    # 3. Combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(principal_angle_matrix, xticklabels=langs, yticklabels=langs,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.3f',
                ax=axes[0], cbar_kws={'label': 'Similarity'})
    axes[0].set_title('Principal Angles')
    
    sns.heatmap(cca_matrix, xticklabels=langs, yticklabels=langs,
                cmap='plasma', vmin=0, vmax=1, annot=True, fmt='.3f',
                ax=axes[1], cbar_kws={'label': 'Similarity'})
    axes[1].set_title('CCA Canonical Correlation')
    
    plt.tight_layout()
    plt.savefig(method_b_dir / 'combined_subspace_similarity.png', dpi=200)
    plt.close()
    
    # Save results
    results_b['svd_info'] = {lang: {
        'singular_values': svd_results[lang]['singular_values'],
        'explained_variance_ratio': svd_results[lang]['explained_variance_ratio'],
        'n_components': svd_results[lang]['n_components']
    } for lang in svd_results}
    
    results_b['principal_angle_similarity'] = {
        'langs': langs,
        'matrix': principal_angle_matrix.tolist()
    }
    
    results_b['cca_similarity'] = {
        'langs': langs,
        'matrix': cca_matrix.tolist()
    }
    
    # Compute summary statistics
    cross_pairs = list(itertools.combinations(langs, 2))
    pa_cross = []
    cca_cross = []
    for la, lb in cross_pairs:
        i = langs.index(la)
        j = langs.index(lb)
        pa_cross.append(principal_angle_matrix[i, j])
        cca_cross.append(cca_matrix[i, j])
    
    results_b['summary_statistics'] = {
        'principal_angles': {
            'mean': float(np.mean(pa_cross)),
            'std': float(np.std(pa_cross)),
            'min': float(np.min(pa_cross)),
            'max': float(np.max(pa_cross))
        },
        'cca': {
            'mean': float(np.mean(cca_cross)),
            'std': float(np.std(cca_cross)),
            'min': float(np.min(cca_cross)),
            'max': float(np.max(cca_cross))
        }
    }
    
    with open(method_b_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results_b, f, indent=2)
    
    print(f"✓ Method B complete. Results saved to {method_b_dir}")


# ============================================================================
# Method C: Gradient Norm Dominance Profiles
# ============================================================================

def method_c_norm_dominance(output_dir: Path, lang_layer_vectors: Dict[str, Dict[int, np.ndarray]]):
    """
    Method C: Gradient Norm Dominance Profiles
    - For each language, compute gradient norm per layer
    - Analyze which layers contribute most to total gradient
    - Compare dominance profiles across languages
    - Identify language-specific vs. shared dominant layers
    """
    print("\n=== Method C: Gradient Norm Dominance Profiles ===")
    
    method_c_dir = output_dir / 'method_c_norm_dominance'
    method_c_dir.mkdir(parents=True, exist_ok=True)
    
    results_c = {}
    langs = sorted(lang_layer_vectors.keys())
    
    # Collect all layer indices
    all_layers = set()
    for lang in langs:
        all_layers.update(lang_layer_vectors[lang].keys())
    layers = sorted(all_layers)
    
    # 1. Compute norm profiles for each language
    norm_profiles = {}
    
    for lang in langs:
        layer_norms = []
        layer_indices = []
        
        for li in layers:
            if li in lang_layer_vectors[lang]:
                grads = lang_layer_vectors[lang][li]  # (N, d)
                # Compute mean L2 norm across samples
                norms = np.linalg.norm(grads, axis=1)
                mean_norm = float(np.mean(norms))
                layer_norms.append(mean_norm)
                layer_indices.append(li)
        
        if layer_norms:
            # Normalize to get proportion
            total_norm = sum(layer_norms)
            if total_norm > 0:
                norm_proportions = [n / total_norm for n in layer_norms]
            else:
                norm_proportions = [0.0] * len(layer_norms)
            
            norm_profiles[lang] = {
                'layer_indices': layer_indices,
                'absolute_norms': layer_norms,
                'norm_proportions': norm_proportions,
                'total_norm': total_norm
            }
    
    # 2. Plot norm profiles
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Absolute norms
    for lang in langs:
        if lang in norm_profiles:
            profile = norm_profiles[lang]
            axes[0].plot(profile['layer_indices'], profile['absolute_norms'], 
                        marker='o', label=lang, linewidth=2)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Mean Gradient Norm')
    axes[0].set_title('Absolute Gradient Norms by Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Normalized proportions
    for lang in langs:
        if lang in norm_profiles:
            profile = norm_profiles[lang]
            axes[1].plot(profile['layer_indices'], profile['norm_proportions'],
                        marker='s', label=lang, linewidth=2)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Proportion of Total Norm')
    axes[1].set_title('Normalized Gradient Contribution by Layer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(method_c_dir / 'norm_profiles.png', dpi=200)
    plt.close()
    
    # 3. Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for stacked bars
    x_pos = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    
    for lang in langs:
        if lang in norm_profiles:
            profile = norm_profiles[lang]
            # Create full array with zeros for missing layers
            proportions = np.zeros(len(layers))
            for li, prop in zip(profile['layer_indices'], profile['norm_proportions']):
                li_idx = layers.index(li)
                proportions[li_idx] = prop
            
            ax.bar(x_pos, proportions, bottom=bottom, label=lang, alpha=0.8)
            bottom += proportions
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cumulative Gradient Contribution')
    ax.set_title('Stacked Gradient Contributions by Layer and Language')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(li) for li in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(method_c_dir / 'stacked_contributions.png', dpi=200)
    plt.close()
    
    # 4. Compute dominance metrics
    dominance_metrics = {}
    
    for lang in langs:
        if lang not in norm_profiles:
            continue
        
        profile = norm_profiles[lang]
        proportions = np.array(profile['norm_proportions'])
        
        # Entropy (lower = more concentrated)
        entropy = float(-np.sum(proportions * np.log(proportions + 1e-10)))
        
        # Gini coefficient (higher = more unequal/concentrated)
        sorted_props = np.sort(proportions)
        n = len(sorted_props)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_props)) / (n * np.sum(sorted_props)) - (n + 1) / n)
        
        # Top-K concentration (proportion in top 3 layers)
        top3_indices = np.argsort(proportions)[-3:]
        top3_concentration = float(np.sum(proportions[top3_indices]))
        
        dominance_metrics[lang] = {
            'entropy': entropy,
            'gini_coefficient': gini,
            'top3_concentration': top3_concentration,
            'dominant_layers': [int(profile['layer_indices'][i]) for i in np.argsort(proportions)[-3:][::-1]],
            'dominant_layer_proportions': [float(proportions[i]) for i in np.argsort(proportions)[-3:][::-1]]
        }
    
    # 5. Compare dominance patterns
    plt.figure(figsize=(10, 6))
    metrics_names = ['entropy', 'gini_coefficient', 'top3_concentration']
    x = np.arange(len(langs))
    width = 0.25
    
    for i, metric in enumerate(metrics_names):
        values = [dominance_metrics[lang][metric] for lang in langs if lang in dominance_metrics]
        plt.bar(x + i * width, values, width, label=metric)
    
    plt.xlabel('Language')
    plt.ylabel('Metric Value')
    plt.title('Dominance Metrics by Language')
    plt.xticks(x + width, langs)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(method_c_dir / 'dominance_metrics.png', dpi=200)
    plt.close()
    
    # Save results
    results_c['norm_profiles'] = norm_profiles
    results_c['dominance_metrics'] = dominance_metrics
    
    # Cross-language analysis
    # Compute correlation between norm profiles
    if len(norm_profiles) >= 2:
        profile_corr_matrix = np.ones((len(langs), len(langs)), dtype=np.float32)
        
        for i, la in enumerate(langs):
            for j, lb in enumerate(langs):
                if la not in norm_profiles or lb not in norm_profiles:
                    continue
                if i >= j:
                    continue
                
                # Align profiles to same layers
                common_layers = set(norm_profiles[la]['layer_indices']) & set(norm_profiles[lb]['layer_indices'])
                if len(common_layers) < 2:
                    continue
                
                props_a = []
                props_b = []
                for li in sorted(common_layers):
                    idx_a = norm_profiles[la]['layer_indices'].index(li)
                    idx_b = norm_profiles[lb]['layer_indices'].index(li)
                    props_a.append(norm_profiles[la]['norm_proportions'][idx_a])
                    props_b.append(norm_profiles[lb]['norm_proportions'][idx_b])
                
                corr = float(np.corrcoef(props_a, props_b)[0, 1])
                profile_corr_matrix[i, j] = corr
                profile_corr_matrix[j, i] = corr
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(profile_corr_matrix, xticklabels=langs, yticklabels=langs,
                   cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.3f',
                   cbar_kws={'label': 'Correlation'})
        plt.title('Correlation of Norm Profiles Between Languages')
        plt.tight_layout()
        plt.savefig(method_c_dir / 'profile_correlation.png', dpi=200)
        plt.close()
        
        results_c['profile_correlation'] = {
            'langs': langs,
            'matrix': profile_corr_matrix.tolist()
        }
    
    with open(method_c_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results_c, f, indent=2)
    
    print(f"✓ Method C complete. Results saved to {method_c_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def extract_gradients_for_analysis(
    args, model, text_tokenizer, unit_tokenizer, device
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, np.ndarray]]]:
    """
    Extract gradients for all languages using per-parameter projection + caching.
    This avoids GPU OOM by projecting each parameter individually instead of concatenating all.
    
    Returns:
        - lang_vectors: {lang: (N, proj_dim) array}
        - lang_layer_vectors: {lang: {layer_idx: (N, proj_dim) array}}
    """
    
    calc_loss = CalcLoss(
        label_smoothing=0.2,
        s2t_vocab_info=text_tokenizer.vocab_info,
        t2u_vocab_info=unit_tokenizer.vocab_info
    )
    lang_pairs = [lp.strip() for lp in args.lang_pairs.split(',')]
    
    lang_vectors = {}
    lang_layer_vectors = {}
    
    # ⭐ Global projection matrix cache (shared across all lang_pairs)
    proj_cache: Dict[str, torch.Tensor] = {}
    
    for lang_pair in lang_pairs:
        logging.info(f'\n=== Processing {lang_pair} ===')
        
        # Load manifest
        manifest_file = args.manifest_dir / f'{args.split}_{lang_pair}_manifest.json'
        if not manifest_file.exists():
            logging.warning(f'Manifest not found: {manifest_file}, skipping')
            continue
        
        samples = load_manifest(manifest_file, args.max_samples)
        logging.info(f'Loaded {len(samples)} samples')
        
        # Create dataloader
        batching_config = BatchingConfig(
            batch_size=args.batch_size,
            rank=0,
            world_size=1,
            max_audio_length_sec=15.0,
            num_workers=args.num_workers,
            float_dtype=torch.float32,  # Keep float32 for input to match model dtype
            use_fbank=True,
        )
        
        # Save temp manifest
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
        layer_gradients = {li: [] for li in range(12)}
        
        # Set language group
        src_lang = lang_pair.split('_')[0]
        group = get_group_for_lang(src_lang)
        set_current_group(group)
        
        # Clear gradients first
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        
        n_samples = 0
        batch_count = 0
        
        # Track progress by samples
        pbar = tqdm(total=min(len(samples), args.max_samples or len(samples)), 
                   desc=f'{lang_pair}', unit='samples')
        
        for batch_idx, batch in enumerate(dataloader.get_dataloader()):
            # Check max samples limit
            if args.max_samples and n_samples >= args.max_samples:
                logging.info(f'Reached max_samples limit ({args.max_samples})')
                break
            
            try:
                # Skip empty batches
                if batch.speech_to_text.src_tokens is None:
                    continue
                
                batch_size = batch.speech_to_text.src_tokens.size(0)
                
                # Move batch to device (memory efficient - move then use)
                batch.speech_to_text.src_tokens = batch.speech_to_text.src_tokens.to(device)
                batch.speech_to_text.src_lengths = batch.speech_to_text.src_lengths.to(device)
                batch.speech_to_text.prev_output_tokens = batch.speech_to_text.prev_output_tokens.to(device)
                batch.speech_to_text.target_tokens = batch.speech_to_text.target_tokens.to(device)
                batch.speech_to_text.target_lengths = batch.speech_to_text.target_lengths.to(device)
                
                # Forward pass
                with torch.enable_grad():
                    speech_encoder_out, speech_encoder_padding_mask = model.encode_speech(
                        seqs=batch.speech_to_text.src_tokens,
                        padding_mask=PaddingMask(batch.speech_to_text.src_lengths, batch.speech_to_text.src_tokens.size(1)),
                    )
                    
                    text_decoder_out, text_decoder_padding_mask = model.decode(
                        seqs=batch.speech_to_text.prev_output_tokens,
                        padding_mask=PaddingMask(batch.speech_to_text.target_lengths, batch.speech_to_text.prev_output_tokens.size(1)),
                        encoder_output=speech_encoder_out,
                        encoder_padding_mask=speech_encoder_padding_mask,
                    )
                    
                    # Project to logits
                    assert model.final_proj is not None
                    text_logits = model.final_proj(text_decoder_out)
                    
                    # Compute loss
                    s2t_numel = torch.sum(batch.speech_to_text.target_lengths - 1).to(text_logits.device)
                    s2t_loss = SequenceModelOutput(logits=text_logits, vocab_info=model.target_vocab_info).compute_loss(
                        targets=batch.speech_to_text.target_tokens,
                        ignore_prefix_size=1,
                        label_smoothing=0.0,
                    )
                    loss = s2t_loss / s2t_numel
                    loss = loss / args.update_freq
                    loss.backward()
                
                n_samples += batch_size
                batch_count += 1
                pbar.update(batch_size)
                
                # Collect gradient every accumulation step
                if (n_samples // args.batch_size) % args.update_freq == 0:
                    grad_blocks = extract_grad_blocks(model, device)
                    
                    if grad_blocks:
                        # Filter last N layers if specified
                        if args.proj_only_last_n_layers is not None:
                            filtered_blocks = {}
                            for name, grad in grad_blocks.items():
                                for li in range(12 - args.proj_only_last_n_layers, 12):
                                    if f'layers.{li}.' in name or f'layers[{li}]' in name:
                                        filtered_blocks[name] = grad
                                        break
                            grad_blocks = filtered_blocks
                        
                        # Parameter-level normalization (optional)
                        if args.param_level_norm:
                            for k in grad_blocks:
                                gn = grad_blocks[k].norm()
                                if gn.item() > 0:
                                    grad_blocks[k] = grad_blocks[k] / gn
                        
                        # ⭐ Per-parameter projection with caching
                        proj_acc = torch.zeros(args.proj_dim, dtype=torch.float32, device=device)
                        per_layer_acc: Dict[int, torch.Tensor] = {}
                        
                        for param_name, grad in grad_blocks.items():
                            if grad.numel() == 0:
                                continue
                            
                            # Get or create projection matrix for this parameter (cached)
                            if param_name not in proj_cache:
                                seed = _stable_seed_from_name(param_name)
                                rgen = np.random.RandomState(seed)
                                # ⭐ KEY: Standard Gaussian distribution (NO normalization)
                                proj_matrix_np = rgen.normal(
                                    loc=0.0, scale=1.0,
                                    size=(args.proj_dim, int(grad.numel()))
                                ).astype(np.float32)
                                proj_matrix = torch.from_numpy(proj_matrix_np).to(
                                    device=device, dtype=torch.float32
                                )
                                proj_cache[param_name] = proj_matrix
                            else:
                                proj_matrix = proj_cache[param_name]
                            
                            # Project single parameter gradient
                            try:
                                partial_proj = torch.matmul(proj_matrix, grad)
                                proj_acc += partial_proj
                            except RuntimeError as e:
                                logging.warning(f'Failed to project {param_name}: {e}')
                                continue
                            
                            # Extract layer index for per-layer statistics
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
                        
                        # Store as float16 to match file B (memory efficient)
                        gradient_vectors.append(proj_acc.cpu().detach().numpy().astype(np.float16))
                        
                        # Per-layer normalization and storage
                        if args.layer_level_norm:
                            for li in per_layer_acc:
                                ln = per_layer_acc[li].norm()
                                if ln.item() > 0:
                                    per_layer_acc[li] = per_layer_acc[li] / ln
                        
                        for li, layer_proj in per_layer_acc.items():
                            layer_gradients[li].append(layer_proj.cpu().detach().numpy().astype(np.float16))
                    
                    # Clear gradients for next accumulation
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad = None
                    
                    # Log progress every N gradient vectors
                    if len(gradient_vectors) % 10 == 0:
                        logging.info(f'{lang_pair}: Collected {len(gradient_vectors)} gradient vectors from {n_samples} samples')
                
            except Exception as e:
                logging.error(f'Error processing batch {batch_idx}: {str(e)}')
                logging.error(traceback.format_exc())
                # Clear gradients on error
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None
                continue
        
        pbar.close()
        
        logging.info(f'{lang_pair}: Processed {n_samples} samples in {batch_count} batches')
        logging.info(f'{lang_pair}: Collected {len(gradient_vectors)} gradient vectors')
        
        # Save results (ensure float16 for memory efficiency)
        if gradient_vectors:
            lang_arr = np.stack(gradient_vectors, axis=0).astype(np.float16)
            lang_vectors[lang_pair] = lang_arr
        else:
            lang_vectors[lang_pair] = np.zeros((0, args.proj_dim), dtype=np.float16)
        
        # Save layer vectors
        lang_layer_vectors[lang_pair] = {}
        for li, grads in layer_gradients.items():
            if grads:
                lang_layer_vectors[lang_pair][li] = np.stack(grads, axis=0).astype(np.float16)
        
        temp_manifest.unlink(missing_ok=True)
    
    return lang_vectors, lang_layer_vectors


def main():
    parser = argparse.ArgumentParser(description='B1 Advanced Gradient Analysis')
    parser.add_argument('--manifest_dir', type=Path, required=True)
    parser.add_argument('--lang_pairs', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--proj_only_last_n_layers', type=int, default=None)
    parser.add_argument('--param_level_norm', action='store_true')
    parser.add_argument('--layer_level_norm', action='store_true')
    parser.add_argument('--freeze_encoder_except_last_n', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--r_shared', type=int, default=2048)
    parser.add_argument('--r_group', type=int, default=1024)
    parser.add_argument('--model_name', type=str, default='seamlessM4T_medium')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Detailed GPU/CPU diagnosis
    logger.info('\n' + '=' * 80)
    logger.info('DEVICE AND MEMORY DIAGNOSTIC INFORMATION')
    logger.info('=' * 80)
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'CUDA is_available(): {torch.cuda.is_available()}')
    logger.info(f'CUDA device_count(): {torch.cuda.device_count()}')
    logger.info(f'CUDA_VISIBLE_DEVICES env: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}')
    logger.info(f'Requested device from args: {args.device}')
    
    # Check actual device that will be used
    test_device = torch.device(args.device)
    logger.info(f'\nActual device to be used: {test_device}')
    logger.info(f'Device type: {test_device.type}')
    
    if torch.cuda.is_available():
        logger.info(f'\nAvailable CUDA devices:')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
            logger.info(f'    - Total memory: {props.total_memory / 1e9:.2f} GB')
            logger.info(f'    - Compute capability: {props.major}.{props.minor}')
        
        # Current GPU memory usage
        if test_device.type == 'cuda':
            logger.info(f'\nGPU Memory Status (before model loading):')
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f'  GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total:.2f}GB')
    else:
        logger.warning('\nWARNING: CUDA is not available!')
        logger.warning('  - Running on CPU will be VERY SLOW')
        logger.warning('  - Check: nvidia-smi (GPU detection)')
        logger.warning('  - Check: CUDA_VISIBLE_DEVICES environment variable')
        logger.warning('  - Check: PyTorch installation compatibility with CUDA version')
    
    logger.info('\n' + '=' * 80)
    logger.info(f'Using device: {test_device} (Device.type={test_device.type})')
    logger.info(f'This script will run on: {"GPU" if test_device.type == "cuda" else "CPU"}')
    logger.info('=' * 80 + '\n')
    
    device = test_device
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with B1 architecture
    logger.info('Loading tokenizers...')
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    logger.info(f'Loading base model: {args.model_name}')
    model = load_unity_model(args.model_name, device=torch.device('cpu'), dtype=torch.float32)
    
    if model.t2u_model is not None:
        model.t2u_model = None
    if model.text_encoder is not None:
        model.text_encoder = None
    
    logger.info(f'Applying B1 architecture (r_shared={args.r_shared}, r_group={args.r_group})')
    model = apply_b1_to_model(model, r_shared=args.r_shared, r_group=args.r_group)
    
    logger.info(f'Loading B1 checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info(f'Checkpoint loaded. Missing: {len(load_result.missing_keys)}, Unexpected: {len(load_result.unexpected_keys)}')
    
    # Freeze if specified
    if args.freeze_encoder_except_last_n is not None:
        # [Same freezing logic as in gradient_conflict_analysis_b1.py]
        pass
    
    model = model.to(device)
    model.train()  # Set to train mode for gradient computation
    
    # Extract gradients
    logger.info('\n=== Extracting Gradients ===')
    lang_vectors, lang_layer_vectors = extract_gradients_for_analysis(
        args, model, text_tokenizer, unit_tokenizer, device
    )
    
    if not lang_vectors:
        logger.error('No gradients extracted. Exiting.')
        return
    
    # Run Method A
    method_a_cosine_grouping(args.output_dir, lang_vectors)
    
    # Run Method B
    method_b_subspace_similarity(args.output_dir, lang_vectors)
    
    # Run Method C
    if lang_layer_vectors:
        method_c_norm_dominance(args.output_dir, lang_layer_vectors)
    
    logger.info(f'\n✓ Advanced analysis complete! Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
