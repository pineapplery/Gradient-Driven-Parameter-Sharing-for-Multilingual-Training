#!/usr/bin/env python3
"""
Gradient conflict analysis for multiple language pairs.

- Reads per-language manifest JSONL files (manifest_dir/{split}_{lang_pair}_manifest.json).
- Loads model checkpoint and wrapper `SeamlessM4TMediumModel`.
- For each language, iterates samples, forms single-language batches, computes loss (S2TT CE), accumulates gradients with `update_freq` and extracts a gradient vector per accumulation step.
- Saves per-language gradient vectors (.npy), computes cosine similarities between languages, outputs histograms and heatmaps, and writes summary JSON.

Usage example:
python /mnt/inspurfs/user-fs/224040284/gradient_conflict_analysis.py \
  --manifest_dir /224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests/gcc \
  --lang_pairs aeb_eng,bem_eng,est_eng,gle_eng \
  --split gcc \
  --checkpoint /224040284/workspace/seamless_communication/src/seamless_communication/output/unify_checkpoint2.pt \
  --output_dir /224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/unify_finetuned \
  --batch_size 4 --update_freq 4 --max_samples 800

Notes:
- Expects manifest JSONL where each line has `source.audio_local_path` (pointing to .npy fbank) and `target.text`.
- Default behavior uses gradient per accumulation; use `--per_sample` to compute one gradient per single sample (slower).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import hashlib
import re
import math
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Use the same unity model and dataloader used in training to ensure feature dims match.
from seamless_communication.models.unity.loader import load_unity_model, load_unity_text_tokenizer, load_unity_unit_tokenizer
from seamless_communication.scripts.dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig
from fairseq2.nn.padding import PaddingMask
from fairseq2.models.sequence import SequenceModelOutput
from seamless_communication.cli.m4t.finetune.trainer import CalcLoss


def load_manifest(manifest_path: Path, max_samples: int = None) -> List[Dict]:
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
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
        return float('nan')
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    if pooled == 0:
        return float('nan')
    return (a.mean() - b.mean()) / pooled


def _stable_seed_from_name(name: str) -> int:
    # deterministic 32-bit int seed from name
    h = hashlib.md5(name.encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'little')


def compute_and_save(sim_out_dir: Path, lang_vectors: Dict[str, np.ndarray], bins=200):
    norms = {k: l2_normalize_rows(v) for k, v in lang_vectors.items()}
    results_summary = {}

    for main_lang, V_main in norms.items():
        out_png = sim_out_dir / f"sim_hist_{main_lang}.png"
        plt.figure(figsize=(8, 6))
        for other_lang, V_other in norms.items():
            if V_main.size == 0 or V_other.size == 0:
                sims = np.array([])
            else:
                sims = (V_main @ V_other.T).ravel()
            counts, bin_edges = np.histogram(sims, bins=bins, range=(-1, 1))
            center = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            plt.plot(center, counts, label=f"{main_lang} vs {other_lang}")
            results_summary.setdefault(main_lang, {})[other_lang] = {
                'mean': float(np.mean(sims)) if sims.size else None,
                'std': float(np.std(sims)) if sims.size else None,
                'count': int(sims.size)
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
        Vi = norms[li]
        for j, lj in enumerate(langs):
            Vj = norms[lj]
            if Vi.size == 0 or Vj.size == 0:
                sims = np.array([])
            else:
                sims = (Vi @ Vj.T).ravel()
            mean_mat[i, j] = float(np.mean(sims)) if sims.size else float('nan')
            std_mat[i, j] = float(np.std(sims)) if sims.size else float('nan')
            count_mat[i, j] = int(sims.size)

    # Save CSVs
    import csv
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
            writer.writerow([li] + [str(count_mat[i,j]) for j in range(n_lang)])

    # Save JSON summary for pairwise
    pairwise_summary = { 'langs': langs, 'mean': mean_mat.tolist(), 'std': std_mat.tolist(), 'count': count_mat.tolist() }
    with open(sim_out_dir / 'pairwise_summary.json', 'w', encoding='utf-8') as f:
        json.dump(pairwise_summary, f, indent=2)

    # Plot 4x4 heatmap of mean similarities
    heat_png = sim_out_dir / 'pairwise_heat_mean.png'
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_mat, xticklabels=langs, yticklabels=langs, cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.3f')
    plt.title('Pairwise mean cosine similarity')
    plt.tight_layout()
    plt.savefig(heat_png, dpi=200)
    plt.close()

    with open(sim_out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)


def run_global_embeddings_visuals(sim_out_dir: Path, lang_vectors: Dict[str, np.ndarray]):
    # PCA + t-SNE on the projected vectors (concatenate langs)
    all_vecs = []
    labels = []
    for lang, arr in lang_vectors.items():
        if arr.size == 0:
            continue
        all_vecs.append(arr.astype(np.float32))
        labels += [lang] * arr.shape[0]
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
        mask = [l == u for l in labels]
        pts = Xts[mask]
        plt.scatter(pts[:, 0], pts[:, 1], label=u, s=6)
    plt.legend()
    plt.title('t-SNE of projected gradient sketches')
    plt.tight_layout()
    plt.savefig(sim_out_dir / 'tsne_proj.png', dpi=200)
    plt.close()

    # PCA first two components
    plt.figure(figsize=(8, 6))
    Xp2 = pca.transform(X)[:, :2]
    for u in uniq:
        mask = [l == u for l in labels]
        pts = Xp2[mask]
        plt.scatter(pts[:, 0], pts[:, 1], label=u, s=6)
    plt.legend()
    plt.title('PCA (2D) of projected gradient sketches')
    plt.tight_layout()
    plt.savefig(sim_out_dir / 'pca_proj.png', dpi=200)
    plt.close()


def compute_and_save_layerwise(sim_out_dir: Path, lang_layer_vectors: Dict[str, Dict[int, np.ndarray]]):
    """
    lang_layer_vectors: { lang: { layer_idx: (N, proj_dim) ndarray } }
    """
    # collect all layer indices
    layer_set = set()
    langs = sorted(lang_layer_vectors.keys())
    for lang, mapping in lang_layer_vectors.items():
        for li in mapping.keys():
            layer_set.add(int(li))
    if not layer_set:
        return
    layers = sorted(layer_set)

    # Prepare results structure
    layer_pairwise = {}
    dist_tests = {}

    for li in layers:
        layer_pairwise[li] = {}
        dist_tests[li] = {}
        # for each lang pair compute sims
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

        # distribution tests (within vs between) for each lang pair
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_dir', type=Path, required=True)
    parser.add_argument('--lang_pairs', type=str, default='aeb_eng,bem_eng,est_eng,gle_eng')
    parser.add_argument('--split', type=str, default='gcc')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model_name', type=str, default='seamlessM4T_medium')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of DataLoader workers for preprocessing')
    parser.add_argument('--per_sample', action='store_true')
    parser.add_argument('--max_samples', type=int, default=800)
    parser.add_argument('--proj_dim', type=int, default=512, help='Random projection dimension for gradient sketching')
    parser.add_argument('--save_full_vectors', action='store_true', help='If set, save full gradient vectors (may be very large)')
    parser.add_argument('--freeze_encoder_except_last_n', type=int, default=0,
                        help='Freeze all parameters except last N layers of speech encoder')
    parser.add_argument('--param_level_norm', action='store_true', help='L2-normalize each parameter block before projection')
    parser.add_argument('--layer_level_norm', action='store_true', help='L2-normalize per-layer projected vector after summing blocks')
    parser.add_argument('--proj_only_last_n_layers', type=int, default=2, help='When projecting, only include last N encoder layers (streaming/block projection)')
    parser.add_argument('--per_layer_analysis', action='store_true', help='Compute per-layer pairwise similarity and plots (requires projecting by layer)')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--max_label_len', type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lang_pairs = [lp.strip() for lp in args.lang_pairs.split(',')]

    print(f"Loading unity model from {args.model_name}")
    # Load tokenizers and model (same as training pipeline)
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float16)
    # Load checkpoint if provided
    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        sd = state['model']
    else:
        sd = state
    try:
        model.load_state_dict(sd, strict=False)
        print("Checkpoint loaded into UnitYModel (strict=False)")
    except Exception as e:
        print("Warning: loading checkpoint raised:", e)
    # Optionally freeze parameters except last N encoder layers (mirror finetune script)
    def freeze_encoder_layers(model, n_last_layers: int):
        print(f"Freezing model except last {n_last_layers} layers of speech encoder")
        # Freeze all params first
        for param in model.parameters():
            param.requires_grad = False

        encoder = getattr(model, 'speech_encoder', None)
        if encoder is None:
            print("Warning: model has no attribute 'speech_encoder' — cannot apply encoder freezing")
            return

        # Attempt to unfreeze last n layers in encoder.inner.layers
        inner = getattr(encoder, 'inner', None)
        if inner is not None and hasattr(inner, 'layers'):
            total_layers = len(inner.layers)
            if n_last_layers <= 0:
                return
            if n_last_layers > total_layers:
                print(f"Requested to unfreeze {n_last_layers} layers, but encoder has {total_layers}; unfreezing all")
                layers_to_unfreeze = inner.layers
            else:
                layers_to_unfreeze = inner.layers[-n_last_layers:]

            for idx, layer in enumerate(layers_to_unfreeze):
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"Unfroze {len(layers_to_unfreeze)} encoder layers")
        else:
            print("Warning: speech_encoder.inner.layers not found; no layers unfrozen")

    if args.freeze_encoder_except_last_n and args.freeze_encoder_except_last_n > 0:
        freeze_encoder_layers(model, args.freeze_encoder_except_last_n)
    model.to(device)
    model.train()

    # Log parameter counts and some trainable param names
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total parameters: {total_params:,}")
    print(f"Model trainable parameters: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    # Print up to 20 trainable parameter names for inspection
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for i, name in enumerate(trainable_names[:20]):
        print(f"  trainable[{i}]: {name}")
    if len(trainable_names) > 20:
        print(f"  ... and {len(trainable_names)-20} more trainable parameter names")

    sim_out_dir = args.output_dir
    vec_dir = sim_out_dir / 'vectors'
    vec_dir.mkdir(parents=True, exist_ok=True)
    lang_vectors = {}
    lang_layer_vectors = {}  # lang -> layer_idx -> ndarray (n_samples, proj_dim)
    # projection sketch variables (created after seeing first full-dim gradient)
    proj_matrix = None
    proj_dim = int(args.proj_dim)
    rng = np.random.RandomState(42)
    # cache projection matrices per-parameter on the run device to avoid repeated allocations
    proj_cache: Dict[str, 'torch.Tensor'] = {}
    # Will hold layer-level param name mapping for projection when needed
    encoder_layer_param_map = {}

    for lang_pair in lang_pairs:
        manifest_path = args.manifest_dir / f"{args.split}_{lang_pair}_manifest.json"
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}, skipping")
            continue
        print(f"Using manifest for {lang_pair}: {manifest_path}")

        # Use the same dataloader used during training to build batches.
        batching_config = BatchingConfig(
            batch_size=args.batch_size,
            rank=0,
            world_size=1,
            max_audio_length_sec=15.0,
            num_workers=args.num_workers,
            float_dtype=torch.float16 if device.type != 'cpu' else torch.float32,
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
        lang_layer_vecs = {}  # layer_idx -> list of projected vectors
        accum_steps = 1 if args.per_sample else args.update_freq
        n_samples = 0

        # clear grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        pbar = tqdm(total=args.max_samples, desc=f"{lang_pair}")

        for batch in data_loader.get_dataloader():
            if n_samples >= args.max_samples:
                break

            # move tensors to device and proper dtype
            if batch.speech_to_text.src_tokens is None:
                continue
            batch.speech_to_text.src_tokens = batch.speech_to_text.src_tokens.to(device)
            batch.speech_to_text.src_lengths = batch.speech_to_text.src_lengths.to(device)
            batch.speech_to_text.prev_output_tokens = batch.speech_to_text.prev_output_tokens.to(device)
            batch.speech_to_text.target_tokens = batch.speech_to_text.target_tokens.to(device)
            batch.speech_to_text.target_lengths = batch.speech_to_text.target_lengths.to(device)

            # forward/backward with accumulation
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

                # compute loss same as training (ignore language token prefix)
                s2t_numel = torch.sum(batch.speech_to_text.target_lengths - 1).to(text_logits.device)
                s2t_loss = SequenceModelOutput(logits=text_logits, vocab_info=model.target_vocab_info).compute_loss(
                    targets=batch.speech_to_text.target_tokens,
                    ignore_prefix_size=1,
                    label_smoothing=0.0,
                )
                loss = s2t_loss / s2t_numel
                loss = loss / accum_steps
                loss.backward()

            n_samples += batch.speech_to_text.src_tokens.size(0)
            pbar.update(batch.speech_to_text.src_tokens.size(0))

            # collect gradient vector every accumulation
            if (n_samples // args.batch_size) % accum_steps == 0:
                blocks = extract_grad_blocks(model, device)
                if len(blocks) == 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad = None
                    continue

                # Prepare encoder layer param map once (uses parameter names)
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

                # determine which params to project: params in selected encoder layers
                params_to_project = []
                if encoder_layer_param_map:
                    for li, names in encoder_layer_param_map.items():
                        params_to_project += names
                else:
                    # fallback: project all trainable params
                    params_to_project = list(blocks.keys())

                # streaming/block-wise projection: accumulate into proj vector on device
                proj_acc_t = torch.zeros(proj_dim, dtype=torch.float32, device=device)
                # also collect per-layer projections if requested (device tensors)
                per_layer_acc_t: Dict[int, 'torch.Tensor'] = {}
                for pname in params_to_project:
                    g_t = blocks[pname]
                    if g_t.numel() == 0:
                        continue
                    # optional param-level normalization
                    if args.param_level_norm:
                        gn = g_t.norm()
                        if gn.item() > 0:
                            g_t = g_t / gn

                    # reuse per-parameter projection matrix cached on device
                    if pname not in proj_cache:
                        seed = _stable_seed_from_name(pname)
                        # generate deterministic projection matrix on CPU then move to device
                        try:
                            rgen = np.random.RandomState(seed)
                            R_np = rgen.normal(loc=0.0, scale=1.0, size=(proj_dim, int(g_t.numel()))).astype(np.float32)
                        except Exception:
                            # skip extremely large blocks defensively
                            continue
                        R_t = torch.from_numpy(R_np).to(device=device, dtype=torch.float32)
                        proj_cache[pname] = R_t
                    else:
                        R_t = proj_cache[pname]

                    # project using torch.matmul on device
                    try:
                        partial_t = torch.matmul(R_t, g_t)
                    except RuntimeError:
                        # in rare cases of mismatched sizes, skip
                        continue
                    proj_acc_t += partial_t

                    # map param to layer index (if present)
                    m = re.search(r"inner\.layers\.(\d+)", pname)
                    if m:
                        li = int(m.group(1))
                        per_layer_acc_t.setdefault(li, torch.zeros(proj_dim, dtype=torch.float32, device=device))
                        per_layer_acc_t[li] += partial_t

                # normalize final projection vector (on device) then move to CPU numpy
                pnorm_t = proj_acc_t.norm()
                if pnorm_t.item() > 0:
                    proj_acc_t = proj_acc_t / pnorm_t
                proj_vecs.append(proj_acc_t.cpu().numpy().astype(np.float16))

                # process per-layer accumulated projections (normalize on device then move to CPU numpy)
                for li, arr_t in per_layer_acc_t.items():
                    if args.layer_level_norm:
                        ln_t = arr_t.norm()
                        if ln_t.item() > 0:
                            arr_t = arr_t / ln_t
                    lang_layer_vecs.setdefault(li, []).append(arr_t.cpu().numpy().astype(np.float16))

                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None

        pbar.close()

        if len(proj_vecs) > 0:
            lang_arr = np.stack(proj_vecs, axis=0).astype(np.float16)
        else:
            lang_arr = np.zeros((0, proj_dim), dtype=np.float16)

        print(f"Collected {lang_arr.shape[0]} projected gradient vectors (dim={lang_arr.shape[1]}) for {lang_pair}")
        # optionally save full (original) gradients - disabled by default because it's large
        if args.save_full_vectors:
            # note: this will attempt to save full-dim vectors if you changed code to keep them
            try:
                np.save(vec_dir / f"grad_vectors_{lang_pair}.npy", lang_arr)
            except Exception:
                pass

        lang_vectors[lang_pair] = lang_arr
        # stack per-layer vectors
        if 'lang_layer_vecs' in locals() and lang_layer_vecs:
            lang_layer_arrays = {}
            for li, lst in lang_layer_vecs.items():
                if len(lst) > 0:
                    lang_layer_arrays[int(li)] = np.stack(lst, axis=0).astype(np.float16)
            if lang_layer_arrays:
                lang_layer_vectors[lang_pair] = lang_layer_arrays
        else:
            # ensure empty entry
            lang_layer_vectors.setdefault(lang_pair, {})

    compute_and_save(sim_out_dir, lang_vectors, bins=200)
    # global embeddings visuals
    try:
        run_global_embeddings_visuals(sim_out_dir, lang_vectors)
    except Exception as e:
        print("Warning: global visuals failed:", e)

    # per-layer analysis if requested
    if args.per_layer_analysis:
        try:
            compute_and_save_layerwise(sim_out_dir, lang_layer_vectors)
        except Exception as e:
            print("Warning: per-layer analysis failed:", e)

    print("Done. Results in", sim_out_dir)


if __name__ == '__main__':
    main()
