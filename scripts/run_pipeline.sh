#!/usr/bin/env bash
# ==============================================================================
# GDPS 4-language canonical pipeline:
#   Data Processing -> UFT -> Gradient Analysis (Conflict + Advanced)
#     -> B1 grouped fine-tuning -> inference -> evaluation
#
# This is a clean, from-scratch script. It does NOT read from / depend on
# train_multilang.sh / train_gcc.sh (those are private working notes and are
# not part of this repo) and does NOT call any other .sh file — every stage
# below invokes the underlying Python script directly. Every training/analysis
# parameter is copied verbatim from the verified-best runs:
#   - Stage 2 (UFT)      -> unify_checkpoint_stage2.pt
#   - Stage 4 (B1, best) -> b1_checkpoint2.pt  (== b1_checkpoint13.pt config,
#                           differs only in eval_steps/log_steps granularity)
#
# Gradient analysis note: `gradient_analysis_advanced.py` internally computes
# Method A/B/C. Method C (per-layer gradient norm dominance) was found
# unreliable and its result generation is disabled directly inside that
# script (see the comment block in its `main()`) — nothing to configure here.
#
# Prerequisites (NOT produced by this script): the raw, per-language TSVs
#   ${DATA}/{train,valid,test}_{aeb,bem,est,gle}_eng.tsv
# with columns [id, audio, tgt_text, ...] where `audio` points at the original
# .wav files. These come from the official corpora (IWSLT22 Tunisian, BIG-C
# Bemba, LoResMT Estonian, IWSLT23 Irish) and are outside the scope of this repo.
#
# Usage:
#   1) Edit the "USER CONFIG" block below (paths only).
#   2) bash scripts/run_pipeline.sh
# ==============================================================================
set -euo pipefail

# ---------------------------- USER CONFIG ------------------------------------
GP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this repo root (GDPS_4lang/)
DATA="/path/to/data_big"                                 # dir containing the raw {split}_{lang}_eng.tsv files
OUT="/path/to/output"                                     # all fbank/manifests/checkpoints/predictions/analysis go here
DEVICE="cuda"
LANG_PAIRS="aeb_eng,bem_eng,est_eng,gle_eng"
# ------------------------------------------------------------------------------

FBANK_DIR="${OUT}/fbank_features"
MANIFEST_DIR="${OUT}/manifests"
REBAL_MANIFEST_DIR="${MANIFEST_DIR}/rebalance_manifest"
GCC_MANIFEST_DIR="${MANIFEST_DIR}/gcc"          # per-language gcc_{lang}_eng_manifest.json files

UNIFY_CKPT="${OUT}/checkpoint/unify/unify_checkpoint_stage2.pt"
B1_CKPT="${OUT}/checkpoint/b1b2_mixed_training/b1_checkpoint2.pt"
PRED_DIR="${OUT}/group_finetuned/B1_mix_2"
CONFLICT_DIR="${OUT}/grad_analysis/conflict"
ADVANCED_DIR="${OUT}/grad_analysis/advanced"
CONFIG_DIR="${OUT}/grad_analysis/config_result"

mkdir -p "$FBANK_DIR" "$MANIFEST_DIR" "$REBAL_MANIFEST_DIR" "$GCC_MANIFEST_DIR" \
         "$(dirname "$UNIFY_CKPT")" "$(dirname "$B1_CKPT")" "$PRED_DIR" \
         "$CONFLICT_DIR" "$ADVANCED_DIR" "$CONFIG_DIR"

echo "================================================================"
echo "[1/6] Data Processing: raw TSV -> fbank .npy -> per-lang manifest -> merged/rebalanced/gcc manifests"
echo "================================================================"

echo "--- [1a] Extract 80-dim fbank features from raw wav TSVs -> *_npy.tsv (written next to the source TSVs) ---"
python "${GP}/scripts/generate_fbank_features.py" \
  --tsv_src "${DATA}" \
  --fbank_out "${FBANK_DIR}" \
  --mode singlelang

echo "--- [1b] TSV (*_npy.tsv) -> per-language, per-split manifest.json ---"
python "${GP}/scripts/prepare_manifest.py" \
  --base_dir "${DATA}" \
  --use_fbank \
  --fbank_dir "${FBANK_DIR}"
# NOTE: prepare_manifest.py writes to ${DATA}/manifests/*.json by convention;
# move/symlink them into ${MANIFEST_DIR} if DATA and OUT are different dirs:
if [ "${DATA}/manifests" != "${MANIFEST_DIR}" ]; then
  cp "${DATA}/manifests/"*_manifest.json "${MANIFEST_DIR}/"
fi

echo "--- [1c] Merge the 4 per-language manifests -> train_all_manifest.json / valid_all_manifest.json ---"
python "${GP}/scripts/merge_manifests.py" \
  --manifest_dir "${MANIFEST_DIR}" \
  --output_dir "${MANIFEST_DIR}" \
  --lang_pairs "${LANG_PAIRS}"

echo "--- [1d] Rebalance the merged manifests (downsample aeb/bem/est towards gle's size) ---"
python "${GP}/scripts/rebalance_manifest.py" \
  --input "${MANIFEST_DIR}/train_all_manifest.json" \
  --output "${REBAL_MANIFEST_DIR}/train_all_manifest.json"
python "${GP}/scripts/rebalance_manifest.py" \
  --input "${MANIFEST_DIR}/valid_all_manifest.json" \
  --output "${REBAL_MANIFEST_DIR}/valid_all_manifest.json"

echo "--- [1e] Prepare per-language 'gcc' split manifests for gradient analysis (Step 3) ---"
# Gradient analysis (gradient_conflict_analysis.py / gradient_analysis_advanced.py)
# expects files named gcc_{lang}_eng_manifest.json. It only samples up to
# --max_samples (800) rows per language regardless of file size, so we simply
# reuse each language's *training* manifest under the 'gcc' split name.
IFS=',' read -ra LP_ARR <<< "${LANG_PAIRS}"
for lp in "${LP_ARR[@]}"; do
  cp "${MANIFEST_DIR}/train_${lp}_manifest.json" "${GCC_MANIFEST_DIR}/gcc_${lp}_manifest.json"
done

echo "================================================================"
echo "[2/6] Stage-1 Unified Fine-Tuning (UFT)  ->  ${UNIFY_CKPT}"
echo "================================================================"
python "${GP}/scripts/finetune_multilang.py" \
  --train_dataset "${MANIFEST_DIR}/train_all_manifest.json" \
  --eval_dataset "${MANIFEST_DIR}/valid_all_manifest.json" \
  --model_name seamlessM4T_medium \
  --save_model_to "${UNIFY_CKPT}" \
  --batch_size 8 \
  --max_epochs 20 \
  --learning_rate 5e-5 \
  --use_fbank \
  --freeze_encoder_except_last_n 2 \
  --device "${DEVICE}" \
  --patience 10 \
  --eval_steps 5000 \
  --warmup_steps 1000 \
  --log_steps 5000

echo "================================================================"
echo "[3/6] Gradient Analysis on UFT checkpoint (Conflict A+B, then Advanced A+B)"
echo "================================================================"

echo "--- [3a] Gradient Conflict analysis (cosine grouping + self/cross purity) ---"
python "${GP}/scripts/gradient_conflict_analysis.py" \
  --manifest_dir "${GCC_MANIFEST_DIR}" \
  --lang_pairs "${LANG_PAIRS}" \
  --split gcc \
  --checkpoint "${UNIFY_CKPT}" \
  --model_name seamlessM4T_medium \
  --output_dir "${CONFLICT_DIR}" \
  --batch_size 4 --update_freq 4 --max_samples 800 --proj_dim 128 \
  --proj_only_last_n_layers 2 --freeze_encoder_except_last_n 2 \
  --param_level_norm --layer_level_norm --per_layer_analysis \
  --num_workers 8

echo "--- [3b] Advanced gradient analysis (Method A: cosine grouping, Method B: SVD+CCA energy; Method C disabled) ---"
python "${GP}/scripts/gradient_analysis_advanced.py" \
  --manifest_dir "${GCC_MANIFEST_DIR}" \
  --lang_pairs "${LANG_PAIRS}" \
  --split gcc \
  --checkpoint "${UNIFY_CKPT}" \
  --model_name seamlessM4T_medium \
  --output_dir "${ADVANCED_DIR}" \
  --batch_size 4 --update_freq 4 --max_samples 800 --proj_dim 128 \
  --proj_only_last_n_layers 2 --freeze_encoder_except_last_n 2 \
  --param_level_norm --layer_level_norm \
  --num_workers 8

echo "--- [3c] Generate B1 group config (group_assignments + share_ratio) from the analysis above ---"
python "${GP}/scripts/generate_config_from_gradient_analysis.py" \
  --kmeans_result "${ADVANCED_DIR}/method_a_cosine_grouping/results.json" \
  --similarity_result "${CONFLICT_DIR}/summary.json" \
  --energy_result "${ADVANCED_DIR}/method_b_subspace_similarity/results.json" \
  --output_dir "${CONFIG_DIR}"

# NOTE: the verified-best b1_checkpoint2/13 config below relies on the
# *hardcoded* LANG_TO_GROUP_2 grouping in models/seamless_m4t_medium/b1_minimal.py
# (aeb/est/gle vs bem), which is exactly the outcome this analysis confirms
# (see README §0.4). The generated ${CONFIG_DIR} config is kept for
# reference/reproducibility of the analysis result, but is not re-fed into
# Stage 4 below to stay faithful to the paper's actual run.

echo "================================================================"
echo "[4/6] Stage-2 B1 grouped fine-tuning (checkpoint2 config)  ->  ${B1_CKPT}"
echo "================================================================"
python "${GP}/scripts/finetune_B1_trainer.py" \
  --train_dataset "${REBAL_MANIFEST_DIR}/train_all_manifest.json" \
  --eval_dataset "${REBAL_MANIFEST_DIR}/valid_all_manifest.json" \
  --model_name seamlessM4T_medium \
  --save_model_to "${B1_CKPT}" \
  --batch_size 4 \
  --max_epochs 20 \
  --device "${DEVICE}" \
  --seed 2343 \
  --eval_steps 5000 \
  --use_fbank \
  --log_steps 5000 \
  --warmup_steps 2000 \
  --patience 15 \
  --load_checkpoint_path "${UNIFY_CKPT}" \
  --learning_rate 4e-5 \
  --group_learning_rate 1e-4 \
  --max_src_tokens 15000 \
  --private_weight_decay 0.05 \
  --dropout_rate 0.05

# NOTE: b1_checkpoint13.pt is the exact same config, only with
# --eval_steps 3000 --log_steps 3000 (finer-grained logging). Uncomment below
# and comment out the block above if you want that variant instead:
#
# python "${GP}/scripts/finetune_B1_trainer.py" \
#   --train_dataset "${REBAL_MANIFEST_DIR}/train_all_manifest.json" \
#   --eval_dataset "${REBAL_MANIFEST_DIR}/valid_all_manifest.json" \
#   --model_name seamlessM4T_medium \
#   --save_model_to "${OUT}/checkpoint/b1b2_mixed_training/b1_checkpoint13.pt" \
#   --batch_size 4 --max_epochs 20 --device "${DEVICE}" --seed 2343 \
#   --eval_steps 3000 --use_fbank --log_steps 3000 --warmup_steps 2000 --patience 15 \
#   --load_checkpoint_path "${UNIFY_CKPT}" \
#   --learning_rate 4e-5 --group_learning_rate 1e-4 --max_src_tokens 15000 \
#   --private_weight_decay 0.05 --dropout_rate 0.05

echo "================================================================"
echo "[5/6] Inference on aeb / bem / est / gle -> eng"
echo "================================================================"
export PYTHONPATH="${GP}/models/seamless_m4t_medium:${PYTHONPATH:-}"

declare -A TEST_TSV=(
  [aeb]="${DATA}/test_aeb_eng_npy.tsv"
  [bem]="${DATA}/test_bem_eng_npy.tsv"
  [est]="${DATA}/test_est_eng_npy.tsv"
  [gle]="${DATA}/test_gle_eng_npy_ori.tsv"   # historical filename uses _ori suffix
)

for lang in aeb bem est gle; do
  python "${GP}/cli/m4t/predict/predict_b1_translator-4lang.py" \
    "${TEST_TSV[$lang]}" \
    --task S2TT \
    --tgt_lang eng \
    --src_lang "${lang}" \
    --load_checkpoint "${B1_CKPT}" \
    --output_path "${PRED_DIR}/output_${lang}_eng_npy_results.tsv"
done

echo "================================================================"
echo "[6/6] Evaluation (BLEU / TER / chrF / chrF++ / BERTScore / COMET)"
echo "================================================================"
python "${GP}/scripts/evaluate_multilang_s2t.py" \
  --ref_dir "${DATA}" \
  --pred_dir "${PRED_DIR}" \
  --output_dir "${PRED_DIR}"

echo "Done. Results: ${PRED_DIR}/all valid evaluation_results.json"
