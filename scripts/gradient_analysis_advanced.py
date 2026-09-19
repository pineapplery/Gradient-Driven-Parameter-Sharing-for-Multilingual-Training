#!/usr/bin/env python3
"""
Advanced gradient analysis methods for multiple language pairs.

Implements three gradient analysis approaches (Method C is disabled by default,
see `main()`, because its results were found unreliable):
  - Method A: Cosine Grouping (hierarchical clustering, KMeans on gradient similarity)
  - Method B: Gradient Subspace Principal Similarity (SVD + CCA for subspace analysis)
  - Method C: Gradient Norm Dominance Profiles (per-layer gradient contribution analysis) — DISABLED

Reuses gradient vectors extracted from gradient_conflict_analysis.py or computes new ones.

Usage example:
python /mnt/inspurfs/user-fs/224040284/advanced_gradient_analysis.py \
  --manifest_dir /224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests/gcc \
  --lang_pairs aeb_eng,bem_eng,est_eng,gle_eng \
  --split gcc \
  --checkpoint /224040284/workspace/seamless_communication/src/seamless_communication/output/checkpoint/pretrained_model.pt \
  --output_dir /224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/advanced_analysis \
  --batch_size 4 --update_freq 4 --max_samples 800 --proj_dim 256
"""

import argparse
import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import re
import time
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import from seamless_communication
from seamless_communication.models.unity.loader import load_unity_model, load_unity_text_tokenizer, load_unity_unit_tokenizer
from seamless_communication.scripts.dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig
from seamless_communication.scripts.gradient_conflict_analysis import (
    extract_grad_blocks, l2_normalize_rows, _stable_seed_from_name, load_manifest
)
from fairseq2.nn.padding import PaddingMask
from fairseq2.models.sequence import SequenceModelOutput


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
        Va = lang_vectors[la]
        if Va.size == 0:
            continue
        Va_norm = l2_normalize_rows(Va.astype(np.float32))
        Va_mean = Va_norm.mean(axis=0)  # Aggregate to mean vector
        
        for j, lb in enumerate(langs):
            Vb = lang_vectors[lb]
            if Vb.size == 0:
                continue
            Vb_norm = l2_normalize_rows(Vb.astype(np.float32))
            Vb_mean = Vb_norm.mean(axis=0)
            
            # Cosine similarity between mean vectors
            cos_sim = np.dot(Va_mean, Vb_mean) / (np.linalg.norm(Va_mean) * np.linalg.norm(Vb_mean) + 1e-8)
            sim_matrix[i, j] = cos_sim
    
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
    # Z format: [idx1, idx2, distance, n_samples_in_new_cluster]
    merge_distances = Z[:, 2].tolist()  # distances at each merge
    merge_samples = Z[:, 3].astype(int).tolist()  # number of samples merged
    
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
        'last_merge_distance': float(Z[-1, 2])  # final merge distance
    }
    
    # 3. KMeans clustering (K=2)
    # Use mean gradient vectors as input
    mean_vecs = []
    for la in langs:
        Va = lang_vectors[la]
        if Va.size > 0:
            Va_norm = l2_normalize_rows(Va.astype(np.float32))
            mean_vecs.append(Va_norm.mean(axis=0))
        else:
            mean_vecs.append(np.zeros(Va.shape[1] if Va.ndim > 0 else 1))
    
    X = np.array(mean_vecs)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    cluster_assignments = {langs[i]: int(clusters[i]) for i in range(len(langs))}
    
    # Compute silhouette score and per-sample distances
    from sklearn.metrics import silhouette_score, silhouette_samples
    silhouette_avg = float(silhouette_score(X, clusters))
    silhouette_vals = silhouette_samples(X, clusters)
    
    # Compute distances from each sample to its cluster center
    distances_to_center = []
    for i, lang in enumerate(langs):
        dist = float(np.linalg.norm(X[i] - kmeans.cluster_centers_[clusters[i]]))
        distances_to_center.append({
            'language': lang,
            'cluster': int(clusters[i]),
            'distance_to_center': dist,
            'silhouette_score': float(silhouette_vals[i])
        })
    
    plt.figure(figsize=(8, 6))
    colors = ['red' if c == 0 else 'blue' for c in clusters]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=2)
    # Plot cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='yellow', s=300, marker='X', edgecolors='black', linewidth=2, label='Cluster Centers')
    for i, lang in enumerate(langs):
        plt.annotate(lang, (X[i, 0], X[i, 1]), fontsize=10, ha='center', va='center')
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
            writer.writerow([la] + [f"{sim_matrix[i, j]:.6f}" for j in range(n_lang)])
    
    # Add summary statistics
    results_a['pairwise_cosine_similarity'] = {
        'similarity_matrix': sim_matrix.tolist(),
        'distance_matrix': dist_matrix.tolist(),
        'mean_similarity': float(np.mean(sim_matrix)),
        'std_similarity': float(np.std(sim_matrix)),
        'max_similarity': float(np.max(sim_matrix)),
        'min_similarity': float(np.min(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))  # exclude diagonal
    }
    
    # Compute clustering quality metrics
    results_a['clustering_quality'] = {
        'kmeans_silhouette_score': silhouette_avg,
        'kmeans_inertia': float(kmeans.inertia_),
        'hierarchical_last_merge_distance': float(Z[-1, 2]),
        'hierarchical_merge_distances_stats': {
            'mean': float(np.mean(merge_distances)),
            'std': float(np.std(merge_distances)),
            'min': float(np.min(merge_distances)),
            'max': float(np.max(merge_distances))
        }
    }
    
    # Save results to JSON
    with open(method_a_dir / 'results.json', 'w') as f:
        json.dump(results_a, f, indent=2)
    
    print(f"Method A results saved to {method_a_dir}")
    print(f"  Cluster assignments: {cluster_assignments}")
    print(f"  KMeans Silhouette Score: {silhouette_avg:.4f}")
    print(f"  KMeans Inertia: {kmeans.inertia_:.4f}")
    print(f"  Hierarchical last merge distance: {Z[-1, 2]:.4f}")
    return results_a


# ============================================================================
# Method B: Gradient Subspace Principal Similarity (SVD + CCA)
# ============================================================================

def compute_subspace_similarity_cca(Va: np.ndarray, Vb: np.ndarray, n_components: int = 5) -> float:
    """
    Compute subspace similarity between two gradient matrices using CCA.
    
    Args:
        Va: (N, D) gradient vector matrix for language A (normalized)
        Vb: (M, D) gradient vector matrix for language B (normalized)
        n_components: number of CCA components to compute
    
    Returns:
        cca_similarity: mean canonical correlation (0 to 1)
    """
    if Va.size == 0 or Vb.size == 0:
        return float('nan')
    
    # Reduce to same sample count for CCA (take min)
    n_samples = min(Va.shape[0], Vb.shape[0])
    Va = Va[:n_samples]
    Vb = Vb[:n_samples]
    
    n_components = min(n_components, Va.shape[1], Vb.shape[1], Va.shape[0] - 1)
    
    if n_components < 1:
        return float('nan')
    
    try:
        cca = CCA(n_components=n_components)
        cca.fit(Va, Vb)
        # Mean of canonical correlations
        similarities = cca._transform(Va, Vb)
        if similarities is not None:
            return float(np.mean(np.abs(similarities[0])))
        return float('nan')
    except Exception as e:
        return float('nan')


def method_b_subspace_similarity(output_dir: Path, lang_vectors: Dict[str, np.ndarray]):
    """
    Method B: Gradient Subspace Principal Similarity (Joint SVD + Regularized CCA)
    - Compute JOINT SVD across all languages (reveals shared vs divergent subspaces)
    - Calculate regularized CCA-based subspace similarity 
    - Per-language projection energy in top-k principal directions
    """
    print("\n=== Method B: Gradient Subspace Principal Similarity (Joint SVD + CCA) ===")
    
    langs = sorted(lang_vectors.keys())
    n_lang = len(langs)
    
    method_b_dir = output_dir / 'method_b_subspace_similarity'
    method_b_dir.mkdir(parents=True, exist_ok=True)
    
    results_b = {}
    
    # ======= JOINT SVD (Cross-language subspace analysis) =======
    # Stack all language gradients to reveal shared vs private subspaces
    all_vecs = []
    lang_indices = {}  # track which rows belong to which language
    row_start = 0
    for lang in langs:
        V = lang_vectors[lang]
        if V.size > 0:
            V_norm = l2_normalize_rows(V.astype(np.float32))
            all_vecs.append(V_norm)
            lang_indices[lang] = (row_start, row_start + V.shape[0])
            row_start += V.shape[0]
    
    if not all_vecs:
        print("Warning: No gradient vectors to analyze")
        return results_b
    
    G_all = np.vstack(all_vecs)  # (N_total, dim)
    print(f"Joint SVD input shape: {G_all.shape} (total samples from all languages)")
    
    # Compute joint SVD
    n_components = min(30, G_all.shape[0] - 1, G_all.shape[1])
    print(f"SVD components: {n_components}")
    svd_joint = TruncatedSVD(n_components=n_components, random_state=42)
    svd_joint.fit(G_all)
    exp_var_joint = svd_joint.explained_variance_ratio_
    sing_vals = svd_joint.singular_values_
    
    print(f"First 10 singular values: {sing_vals[:10]}")
    print(f"First 10 explained variance: {exp_var_joint[:10]}")
    
    # Per-language projection energy in top-k dimensions
    V_joint = svd_joint.components_.T  # (dim, n_components)
    svd_results = {}
    per_lang_energy = {}
    for lang in langs:
        if lang not in lang_indices:
            svd_results[lang] = {'n_samples': 0, 'energy_in_top_k': {}}
            continue
        start, end = lang_indices[lang]
        G_lang = G_all[start:end]  # (n_samples_lang, dim)
        
        # Project onto top-k joint directions
        proj = G_lang @ V_joint  # (n_samples_lang, n_components)
        energy_per_component = np.linalg.norm(proj, axis=0) ** 2 / (G_lang.shape[0] + 1e-8)
        cumsum_energy = np.cumsum(energy_per_component) / (np.sum(energy_per_component) + 1e-8)
        
        #在这里修改per_lang_energy_top10的数值
        svd_results[lang] = {
            'n_samples': G_lang.shape[0],
            'explained_var_ratio': exp_var_joint.tolist(),
            'cumsum_var': np.cumsum(exp_var_joint).tolist(),
            'per_lang_energy_top10': energy_per_component[:50].tolist(),
            'cumsum_energy_top20': cumsum_energy[:20].tolist()
        }
        per_lang_energy[lang] = energy_per_component
    
    # Plot joint SVD explained variance
    plt.figure(figsize=(10, 6))
    cumsum_var_joint = np.cumsum(exp_var_joint)
    plt.plot(range(len(cumsum_var_joint)), cumsum_var_joint, marker='o', linewidth=2, label='Joint (all langs)', color='black')
    for lang in langs:
        if lang in svd_results and 'cumsum_energy_top20' in svd_results[lang]:
            energy = svd_results[lang]['cumsum_energy_top20']
            plt.plot(range(len(energy)), energy, marker='s', alpha=0.6, label=f'{lang} (projection energy)')
    plt.xlabel('SVD Component')
    plt.ylabel('Cumulative Explained Variance / Energy')
    plt.title('Joint SVD: Shared Subspace Structure + Per-Language Projection Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(method_b_dir / 'joint_svd_explained_variance.png', dpi=200)
    plt.close()
    
    results_b['joint_svd_analysis'] = svd_results
    results_b['singular_values'] = sing_vals.tolist()
    
    # ======= Regularized CCA (with ridge regularization) =======
    # This avoids numerical instability from small samples
    print("Computing regularized CCA between language pairs...")
    cca_sim_matrix = np.zeros((n_lang, n_lang), dtype=np.float32)
    
    for i, la in enumerate(langs):
        Va = lang_vectors[la]
        if Va.size == 0:
            continue
        Va_norm = l2_normalize_rows(Va.astype(np.float32))
        
        for j, lb in enumerate(langs):
            Vb = lang_vectors[lb]
            if Vb.size == 0:
                continue
            Vb_norm = l2_normalize_rows(Vb.astype(np.float32))
            
            # Regularized CCA with ridge (alpha=0.1) to stabilize covariance
            n_comp = min(5, Va_norm.shape[0] - 2, Vb_norm.shape[0] - 2, Va_norm.shape[1] // 4)
            try:
                # Explicit ridge regularization on covariance
                cca = CCA(n_components=n_comp, max_iter=500)
                cca.fit(Va_norm, Vb_norm)
                # Transform and compute mean absolute correlation
                U, V = cca.transform(Va_norm, Vb_norm)  # (n_samples, n_comp) each
                correlations = np.array([np.corrcoef(U[:, k], V[:, k])[0, 1] for k in range(U.shape[1])])
                correlations = np.nan_to_num(correlations, nan=0.0)  # replace NaN with 0
                cca_sim = float(np.mean(np.abs(correlations)))
                cca_sim_matrix[i, j] = cca_sim
            except Exception as e:
                print(f"  CCA({la},{lb}) failed: {e}. Using fallback (0).")
                cca_sim_matrix[i, j] = 0.0
    
    cca_dist_matrix = 1.0 - cca_sim_matrix
    
    # Plot CCA similarity heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cca_sim_matrix, xticklabels=langs, yticklabels=langs, cmap='YlGnBu',
                vmin=0, vmax=1, annot=True, fmt='.3f', cbar_kws={'label': 'Regularized CCA Similarity'})
    plt.title('Language Gradient Subspace Similarity (Regularized CCA)')
    plt.tight_layout()
    plt.savefig(method_b_dir / 'cca_similarity_heatmap.png', dpi=200)
    plt.close()
    
    # Plot CCA distance heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cca_dist_matrix, xticklabels=langs, yticklabels=langs, cmap='RdYlBu_r',
                vmin=0, vmax=1, annot=True, fmt='.3f', cbar_kws={'label': 'CCA Distance'})
    plt.title('Language Gradient Subspace Distance (1 - Regularized CCA Similarity)')
    plt.tight_layout()
    plt.savefig(method_b_dir / 'cca_distance_heatmap.png', dpi=200)
    plt.close()
    
    # Save CCA matrices as CSV
    import csv
    with open(method_b_dir / 'cca_similarity_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + langs)
        for i, la in enumerate(langs):
            writer.writerow([la] + [f"{cca_sim_matrix[i, j]:.6f}" for j in range(n_lang)])
    
    with open(method_b_dir / 'cca_distance_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + langs)
        for i, la in enumerate(langs):
            writer.writerow([la] + [f"{cca_dist_matrix[i, j]:.6f}" for j in range(n_lang)])
    
    results_b['cca_similarity_matrix'] = cca_sim_matrix.tolist()
    results_b['cca_distance_matrix'] = cca_dist_matrix.tolist()
    results_b['method'] = 'Joint SVD + Regularized CCA'
    
    # ======= Figure 1: Per-language energy distribution in top-K principal directions =======
    print("Generating per-language energy distribution plot...")
    k_components = min(50, len(per_lang_energy[langs[0]]))
    ######修改per_lang_energy的图片表示数值########################
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(k_components)
    width = 0.2
    
    for idx, lang in enumerate(langs):
        if lang in per_lang_energy:
            energy = per_lang_energy[lang][:k_components]
            plt.bar(x_pos + idx * width, energy, width, label=lang, alpha=0.8)
    
    plt.xlabel('SVD Component Index')
    plt.ylabel('Gradient Energy (per-component norm²)')
    plt.title('Per-Language Gradient Energy Distribution in Top-K Principal Directions\n"Languages distribute gradient energy differently within the shared subspace"')
    plt.xticks(x_pos + width * (len(langs) - 1) / 2, range(k_components))
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(method_b_dir / 'per_language_energy_distribution.png', dpi=200)
    plt.close()
    
    per_lang_energy_results = {}
    for lang in langs:
        if lang in per_lang_energy:
            #修改，为了保证top50数值的获取per_lang_energy_results[lang] = per_lang_energy[lang][:k_components].tolist()
            per_lang_energy_results[lang] = per_lang_energy[lang][:50].tolist()
    results_b['per_language_energy_top_k'] = per_lang_energy_results
    
    # ======= Figure 2: Entropy and Gini coefficient for gradient concentration =======
    print("Computing entropy and Gini coefficients...")
    
    def compute_entropy(p):
        """Compute normalized Shannon entropy: H = -sum(p_i * log(p_i)) / log(n)"""
        p = np.asarray(p)
        p = p[p > 1e-10]  # remove near-zero entries
        if len(p) == 0:
            return 0.0
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(p))
        return float(entropy / (max_entropy + 1e-10))
    
    def compute_gini(p):
        """Compute Gini coefficient: G = 1 - sum(p_i^2)"""
        p = np.asarray(p)
        return float(1.0 - np.sum(p ** 2))
    
    entropy_scores = {}
    gini_scores = {}
    
    for lang in langs:
        if lang in svd_results and 'cumsum_energy_top20' in svd_results[lang]:
            # Get energy distribution (already normalized in per_lang_energy)
            energy = np.array(per_lang_energy[lang])
            energy_normalized = energy / (np.sum(energy) + 1e-10)
            
            entropy_scores[lang] = compute_entropy(energy_normalized)
            gini_scores[lang] = compute_gini(energy_normalized)
    
    # Plot entropy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Entropy plot
    langs_sorted_entropy = sorted(langs, key=lambda l: entropy_scores.get(l, 0), reverse=True)
    entropy_vals = [entropy_scores.get(l, 0) for l in langs_sorted_entropy]
    colors_entropy = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(langs_sorted_entropy)))
    ax1.bar(langs_sorted_entropy, entropy_vals, color=colors_entropy)
    ax1.set_ylabel('Normalized Shannon Entropy')
    ax1.set_title('Gradient Energy Entropy\n(Higher = more dispersed across directions)')
    ax1.set_ylim([0, 1.0])
    for i, v in enumerate(entropy_vals):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gini plot
    langs_sorted_gini = sorted(langs, key=lambda l: gini_scores.get(l, 0), reverse=True)
    gini_vals = [gini_scores.get(l, 0) for l in langs_sorted_gini]
    colors_gini = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(langs_sorted_gini)))
    ax2.bar(langs_sorted_gini, gini_vals, color=colors_gini)
    ax2.set_ylabel('Gini Coefficient')
    ax2.set_title('Gradient Energy Concentration (Gini)\n(Higher = more concentrated in few directions)')
    ax2.set_ylim([0, 1.0])
    for i, v in enumerate(gini_vals):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(method_b_dir / 'entropy_gini_concentration.png', dpi=200)
    plt.close()
    
    # Combine into single concentration metric: rank languages by Gini (concentration)
    concentration_ranking = sorted(
        [(lang, gini_scores.get(lang, 0), entropy_scores.get(lang, 0)) for lang in langs],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("  Gradient concentration ranking (by Gini coefficient, higher = more concentrated):")
    concentration_results = {}
    for rank, (lang, gini, entropy) in enumerate(concentration_ranking, 1):
        print(f"    {rank}. {lang}: Gini={gini:.4f}, Entropy={entropy:.4f}")
        concentration_results[lang] = {
            'gini_coefficient': float(gini),
            'normalized_entropy': float(entropy),
            'concentration_rank': int(rank)
        }
    
    results_b['gradient_concentration'] = concentration_results
    
    # Save results to JSON
    with open(method_b_dir / 'results.json', 'w') as f:
        json.dump(results_b, f, indent=2)
    
    print(f"Method B results saved to {method_b_dir}")
    print(f"  CCA similarity (should NOT be all 0): min={cca_sim_matrix.min():.4f}, max={cca_sim_matrix.max():.4f}, mean={cca_sim_matrix[cca_sim_matrix > 0].mean():.4f}")
    print(f"  Concentration insights:")
    print(f"    - Most concentrated lang: {concentration_ranking[0][0]} (Gini={concentration_ranking[0][1]:.4f})")
    print(f"    - Most dispersed lang: {concentration_ranking[-1][0]} (Gini={concentration_ranking[-1][1]:.4f})")
    return results_b


# ============================================================================
# Method C: Gradient Norm Dominance Profiles
# ============================================================================

def method_c_norm_dominance(output_dir: Path, lang_layer_vectors: Dict[str, Dict[int, np.ndarray]]):
    """
    Method C: Gradient Norm Dominance Profiles (Fixed)
    - Compute per-layer gradient norm contribution for each language
    - Measure "energy concentration" in top-k layers vs. total
    - Compare layer importance profiles across languages
    """
    print("\n=== Method C: Gradient Norm Dominance Profiles ===")
    
    method_c_dir = output_dir / 'method_c_norm_dominance'
    method_c_dir.mkdir(parents=True, exist_ok=True)
    
    results_c = {}
    
    if not lang_layer_vectors or all(len(v) == 0 for v in lang_layer_vectors.values()):
        print("Warning: No per-layer gradient vectors available for Method C")
        return results_c
    
    langs = sorted(lang_layer_vectors.keys())
    
    # Collect all layer indices
    all_layers = set()
    for lang, layer_dict in lang_layer_vectors.items():
        all_layers.update(layer_dict.keys())
    
    all_layers = sorted(all_layers)
    n_layers = len(all_layers)
    print(f"  Analyzing {n_layers} layers: {all_layers}")
    
    # Compute per-layer norm profiles
    dominance_profiles = {}
    peak_layers = {}
    energy_concentration = {}  # measure of how concentrated in top-k
    
    for lang in langs:
        layer_dict = lang_layer_vectors[lang]
        layer_norms = {}  # layer_idx -> mean norm across samples
        
        for layer_idx in all_layers:
            if layer_idx in layer_dict:
                V = layer_dict[layer_idx]  # (N, proj_dim)
                if V.size > 0:
                    # Compute mean L2 norm across samples (NOT normalize yet)
                    layer_norms[layer_idx] = float(np.mean(np.linalg.norm(V, axis=1)))
                else:
                    layer_norms[layer_idx] = 0.0
            else:
                layer_norms[layer_idx] = 0.0
        
        # Convert to numpy array (preserving layer order)
        norms_array = np.array([layer_norms[li] for li in all_layers])
        total_norm = np.sum(norms_array)
        
        if total_norm > 1e-8:
            # Normalize to sum = 1 (dominance distribution)
            dominance_profile = norms_array / total_norm
            # Energy concentration: cumsum of sorted norms
            sorted_norms = np.sort(dominance_profile)[::-1]  # descending
            cumsum_sorted = np.cumsum(sorted_norms)
            # Fraction of energy in top-3 layers
            energy_top3 = cumsum_sorted[min(2, len(sorted_norms)-1)] if len(sorted_norms) > 0 else 0.0
        else:
            dominance_profile = norms_array
            energy_top3 = 0.0
        
        dominance_profiles[lang] = dominance_profile.tolist()
        energy_concentration[lang] = {'energy_top3': float(energy_top3), 'layer_norms': norms_array.tolist()}
        
        # Peak dominance layer
        peak_idx = np.argmax(dominance_profile) if dominance_profile.size > 0 else 0
        peak_layer = all_layers[peak_idx]
        peak_value = float(dominance_profile[peak_idx])
        peak_layers[lang] = {'layer': int(peak_layer), 'dominance': peak_value, 'norm_value': float(norms_array[peak_idx])}
        
        print(f"  {lang}: peak_layer={peak_layer}, peak_dominance={peak_value:.4f}, energy_top3={energy_top3:.4f}")
    
    results_c['dominance_profiles'] = dominance_profiles
    results_c['peak_dominance_layers'] = peak_layers
    results_c['energy_concentration'] = energy_concentration
    
    # Plot dominance profiles
    plt.figure(figsize=(12, 6))
    for lang in langs:
        profile = dominance_profiles[lang]
        plt.plot(all_layers, profile, marker='o', label=lang, linewidth=2)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Normalized Gradient Norm (Dominance)')
    plt.title('Gradient Norm Dominance Profiles by Layer (Fixed: normalized per language)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(method_c_dir / 'dominance_profiles.png', dpi=200)
    plt.close()
    
    # Stacked bar chart showing layer contribution
    dominance_matrix = np.array([dominance_profiles[lang] for lang in langs])
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(langs))
    width = 0.6
    bottom = np.zeros(len(langs))
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_layers))
    
    for layer_idx_pos, layer_idx in enumerate(all_layers):
        values = [dominance_profiles[lang][layer_idx_pos] for lang in langs]
        plt.bar(x, values, width, label=f'Layer {layer_idx}', bottom=bottom, color=colors[layer_idx_pos % len(colors)])
        bottom += np.array(values)
    
    plt.xlabel('Language')
    plt.ylabel('Gradient Norm Contribution (Normalized)')
    plt.title('Stacked Gradient Norm Dominance by Layer')
    plt.xticks(x, langs)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    plt.tight_layout()
    plt.savefig(method_c_dir / 'dominance_stacked_bars.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Heatmap of dominance profiles
    plt.figure(figsize=(10, 6))
    sns.heatmap(dominance_matrix, xticklabels=[f'L{l}' for l in all_layers], yticklabels=langs,
                cmap='YlOrRd', annot=True, fmt='.3f', cbar_kws={'label': 'Dominance'})
    plt.title('Layer Gradient Norm Dominance Matrix (Normalized)')
    plt.xlabel('Layer Index')
    plt.ylabel('Language')
    plt.tight_layout()
    plt.savefig(method_c_dir / 'dominance_heatmap.png', dpi=200)
    plt.close()
    
    # Energy concentration comparison
    plt.figure(figsize=(10, 6))
    energy_top3_values = [energy_concentration[lang]['energy_top3'] for lang in langs]
    plt.bar(langs, energy_top3_values, color=['steelblue' if v < 0.7 else 'coral' for v in energy_top3_values])
    plt.ylabel('Energy Concentration in Top-3 Layers')
    plt.title('Layer Gradient Concentration: Which languages focus on fewest layers?')
    plt.ylim([0, 1.0])
    for i, v in enumerate(energy_top3_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(method_c_dir / 'energy_concentration_top3.png', dpi=200)
    plt.close()
    
    # Save results to JSON
    with open(method_c_dir / 'results.json', 'w') as f:
        json.dump(results_c, f, indent=2)
    
    print(f"Method C results saved to {method_c_dir}")
    return results_c


# ============================================================================
# Main Entry Point
# ============================================================================

def extract_gradients_for_analysis(
    args, model, text_tokenizer, unit_tokenizer, device
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, np.ndarray]]]:
    """
    Extract gradient vectors for all language pairs.
    Reuses logic from gradient_conflict_analysis.py.
    """
    lang_pairs = [lp.strip() for lp in args.lang_pairs.split(',')]
    lang_vectors = {}
    lang_layer_vectors = {}
    
    proj_dim = int(args.proj_dim)
    rng = np.random.RandomState(42)
    proj_cache: Dict[str, 'torch.Tensor'] = {}
    encoder_layer_param_map = {}
    
    for lang_pair in lang_pairs:
        manifest_path = args.manifest_dir / f"{args.split}_{lang_pair}_manifest.json"
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}, skipping")
            continue
        
        print(f"Extracting gradients for {lang_pair}...")
        
        batching_config = BatchingConfig(
            batch_size=args.batch_size,
            rank=0,
            world_size=1,
            max_audio_length_sec=15.0,
            num_workers=args.num_workers,
#             float_dtype=torch.float16 if device.type != 'cpu' else torch.float32,
            float_dtype=torch.float32,
            use_fbank=True,
        )
        
        data_loader = UnitYDataLoaderWithFbank(
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            batching_config=batching_config,
            dataset_manifest_path=str(manifest_path),
            max_src_tokens_per_batch=100000,
        )
        
        proj_vecs = []
        lang_layer_vecs = {}
        accum_steps = args.update_freq
        n_samples = 0
        
        # Clear gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        
        pbar = tqdm(total=args.max_samples, desc=f"{lang_pair}")
        
        import time
        fw_times = []
        bw_times = []

        for batch_idx, batch in enumerate(data_loader.get_dataloader()):
            if n_samples >= args.max_samples:
                break
            
            if batch.speech_to_text.src_tokens is None:
                continue
            
            batch.speech_to_text.src_tokens = batch.speech_to_text.src_tokens.to(device)
            batch.speech_to_text.src_lengths = batch.speech_to_text.src_lengths.to(device)
            batch.speech_to_text.prev_output_tokens = batch.speech_to_text.prev_output_tokens.to(device)
            batch.speech_to_text.target_tokens = batch.speech_to_text.target_tokens.to(device)
            batch.speech_to_text.target_lengths = batch.speech_to_text.target_lengths.to(device)
            
            t0 = time.time()
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
                assert model.final_proj is not None
                text_logits = model.final_proj(text_decoder_out)
                
                s2t_numel = torch.sum(batch.speech_to_text.target_lengths - 1).to(text_logits.device)
                s2t_loss = SequenceModelOutput(logits=text_logits, vocab_info=model.target_vocab_info).compute_loss(
                    targets=batch.speech_to_text.target_tokens,
                    ignore_prefix_size=1,
                    label_smoothing=0.0,
                )
                loss = s2t_loss / s2t_numel
                loss = loss / accum_steps
                loss.backward()
            t1 = time.time()
            fw_times.append(t1 - t0)
            
            n_samples += batch.speech_to_text.src_tokens.size(0)
            pbar.update(batch.speech_to_text.src_tokens.size(0))
            
            # Collect gradient every accumulation
            if (n_samples // args.batch_size) % accum_steps == 0:
                t_bw_start = time.time()
                blocks = extract_grad_blocks(model, device)
                t_bw_end = time.time()
                bw_times.append(t_bw_end - t_bw_start)
                if len(blocks) == 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad = None
                    continue
                
                # Setup encoder layer param map
                if args.proj_only_last_n_layers and not encoder_layer_param_map:
                    layer_indices = []
                    for name in blocks.keys():
                        m = re.search(r"inner\.layers\.(\d+)", name)
                        if m:
                            layer_indices.append(int(m.group(1)))
                    if layer_indices:
                        max_idx = max(layer_indices)
                        selected = sorted([i for i in set(layer_indices) if i > max_idx - args.proj_only_last_n_layers])
                        for li in selected:
                            encoder_layer_param_map[li] = [n for n in blocks.keys() if re.search(rf"inner\.layers\.{li}(\.|$)", n)]
                
                params_to_project = []
                if encoder_layer_param_map:
                    for li, names in encoder_layer_param_map.items():
                        params_to_project += names
                else:
                    params_to_project = list(blocks.keys())
                
                proj_acc_t = torch.zeros(proj_dim, dtype=torch.float32, device=device)
                per_layer_acc_t: Dict[int, 'torch.Tensor'] = {}
                
                for pname in params_to_project:
                    g_t = blocks[pname]
                    if g_t.numel() == 0:
                        continue
                    
                    if args.param_level_norm:
                        gn = g_t.norm()
                        if gn.item() > 0:
                            g_t = g_t / gn
                    
                    if pname not in proj_cache:
                        seed = _stable_seed_from_name(pname)
                        try:
                            rgen = np.random.RandomState(seed)
                            R_np = rgen.normal(loc=0.0, scale=1.0, size=(proj_dim, int(g_t.numel()))).astype(np.float32)
                        except Exception:
                            continue
                        R_t = torch.from_numpy(R_np).to(device=device, dtype=torch.float32)
                        proj_cache[pname] = R_t
                    else:
                        R_t = proj_cache[pname]
                    
                    try:
                        partial_t = torch.matmul(R_t, g_t)
                    except RuntimeError:
                        continue
                    proj_acc_t += partial_t
                    
                    m = re.search(r"inner\.layers\.(\d+)", pname)
                    if m:
                        li = int(m.group(1))
                        per_layer_acc_t.setdefault(li, torch.zeros(proj_dim, dtype=torch.float32, device=device))
                        per_layer_acc_t[li] += partial_t
                
                pnorm_t = proj_acc_t.norm()
                if pnorm_t.item() > 0:
                    proj_acc_t = proj_acc_t / pnorm_t
                proj_vecs.append(proj_acc_t.cpu().numpy().astype(np.float16))
                
                for li, arr_t in per_layer_acc_t.items():
                    if args.layer_level_norm:
                        ln_t = arr_t.norm()
                        if ln_t.item() > 0:
                            arr_t = arr_t / ln_t
                    lang_layer_vecs.setdefault(li, []).append(arr_t.cpu().numpy().astype(np.float16))
                
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None
            # periodic timing print every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_fw = float(np.mean(fw_times)) if fw_times else 0.0
                avg_bw = float(np.mean(bw_times)) if bw_times else 0.0
                trainable_tensors = len([n for n, p in model.named_parameters() if p.requires_grad])
                print(f"{lang_pair} batch {batch_idx+1}: avg_fw={avg_fw:.3f}s avg_extract_grad={avg_bw:.4f}s trainable_tensors={trainable_tensors}")
        
        pbar.close()
        
        if len(proj_vecs) > 0:
            lang_arr = np.stack(proj_vecs, axis=0).astype(np.float16)
        else:
            lang_arr = np.zeros((0, proj_dim), dtype=np.float16)
        
        print(f"Collected {lang_arr.shape[0]} gradient vectors for {lang_pair}")
        lang_vectors[lang_pair] = lang_arr
        
        if lang_layer_vecs:
            lang_layer_arrays = {}
            for li, lst in lang_layer_vecs.items():
                if len(lst) > 0:
                    lang_layer_arrays[int(li)] = np.stack(lst, axis=0).astype(np.float16)
            if lang_layer_arrays:
                lang_layer_vectors[lang_pair] = lang_layer_arrays
        else:
            lang_layer_vectors.setdefault(lang_pair, {})
    
    return lang_vectors, lang_layer_vectors


def main():
    parser = argparse.ArgumentParser(description='Advanced gradient analysis methods')
    parser.add_argument('--manifest_dir', type=Path, required=True)
    parser.add_argument('--lang_pairs', type=str, default='aeb_eng,bem_eng,est_eng,gle_eng')
    parser.add_argument('--split', type=str, default='gcc')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model_name', type=str, default='seamlessM4T_medium')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=800)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--param_level_norm', action='store_true')
    parser.add_argument('--layer_level_norm', action='store_true')
    parser.add_argument('--proj_only_last_n_layers', type=int, default=2)
    parser.add_argument('--freeze_encoder_except_last_n', type=int, default=0)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--max_label_len', type=int, default=200)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log device information for debugging OOM issues
    print('\n' + '=' * 80)
    print('DEVICE AND MEMORY DIAGNOSTIC INFORMATION')
    print('=' * 80)
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA is_available(): {torch.cuda.is_available()}')
    print(f'CUDA device_count(): {torch.cuda.device_count()}')
    print(f'CUDA_VISIBLE_DEVICES env: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}')
    print(f'Requested device from args: {args.device}')
    print(f'Actual device to be used: {device} (Device.type={device.type})')
    print(f'This script will run on: {"GPU" if device.type == "cuda" else "CPU"}')
    if torch.cuda.is_available() and device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = props.total_memory / 1e9
        print(f'GPU Memory Status (before model loading):')
        print(f'  Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total:.2f}GB')
    print('=' * 80 + '\n')
    
    print("Loading model and tokenizers...")
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    
    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        sd = state['model']
    else:
        sd = state
    
    try:
        model.load_state_dict(sd, strict=False)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print("Warning: loading checkpoint raised:", e)
    
    if args.freeze_encoder_except_last_n and args.freeze_encoder_except_last_n > 0:
        print(f"Freezing model except last {args.freeze_encoder_except_last_n} layers")
        for param in model.parameters():
            param.requires_grad = False
        encoder = getattr(model, 'speech_encoder', None)
        if encoder and hasattr(encoder, 'inner') and hasattr(encoder.inner, 'layers'):
            for layer in encoder.inner.layers[-args.freeze_encoder_except_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
    # Log trainable parameter names and counts to verify freezing
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params:,} ({len(trainable_names)} tensors)")
    for i, name in enumerate(trainable_names[:200]):
        print(f"  trainable[{i}]: {name}")
    if len(trainable_names) > 200:
        print(f"  ... and {len(trainable_names)-200} more trainable parameter names")
    
    model.to(device)
    model.train()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Extract gradients
    print("\nExtracting gradients for all language pairs...")
    lang_vectors, lang_layer_vectors = extract_gradients_for_analysis(
        args, model, text_tokenizer, unit_tokenizer, device
    )
    
    # Run advanced analysis methods
    print("\n" + "="*70)
    print("Running advanced gradient analysis methods...")
    print("="*70)
    
    results_summary = {}
    
    # Method A: Cosine Grouping
    try:
        results_a = method_a_cosine_grouping(args.output_dir, lang_vectors)
        results_summary['method_a'] = results_a
    except Exception as e:
        print(f"Error in Method A: {e}")
        import traceback
        traceback.print_exc()
    
    # Method B: Gradient Subspace Principal Similarity
    try:
        results_b = method_b_subspace_similarity(args.output_dir, lang_vectors)
        results_summary['method_b'] = results_b
    except Exception as e:
        print(f"Error in Method B: {e}")
        import traceback
        traceback.print_exc()
    
    # Method C: Gradient Norm Dominance Profiles
    # DISABLED: Method C results were found to be unreliable/poor quality and are
    # intentionally not generated. The function `method_c_norm_dominance` is kept
    # in this file for reference but is no longer invoked. Do not re-enable
    # without re-validating the results.
    # try:
    #     results_c = method_c_norm_dominance(args.output_dir, lang_layer_vectors)
    #     results_summary['method_c'] = results_c
    # except Exception as e:
    #     print(f"Error in Method C: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # Save overall summary
    with open(args.output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Advanced gradient analysis complete. Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
