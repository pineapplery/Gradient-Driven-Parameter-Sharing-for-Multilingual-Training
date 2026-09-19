# GDPS for Interspeech 2026

This directory contains the code for the **Interspeech 2026 paper** *"Automated Gradient-Driven Parameter Sharing for Low-Resource Multilingual Speech-to-Text Translation"*, supporting 4 languages: **aeb, bem, est, gle → eng**.

The backbone is **`seamlessM4T_medium`**. The core idea of GDPS (Gradient-Driven Parameter Sharing) is: first jointly fine-tune the model on multilingual data (UFT), then use gradient statistics to decompose a bottleneck layer in the encoder into a "shared branch + per-group branch" — languages that are similar share parameters, while more distant languages get their own dedicated branch.

---

## Environment Setup

Developed and validated with **Python 3.10 + CUDA 12.1 + PyTorch 2.2.2**. Exact dependency versions are pinned in [`requirements.txt`](requirements.txt).

**1. Create the environment**
```bash
conda create -n gdps python=3.10 -y
conda activate gdps
```

**2. Install Python dependencies** (this also pulls the CUDA 12.1 build of `torch`/`torchaudio` via the extra index URL declared in the file)
```bash
python -m pip install -r requirements.txt
```

**3. Install `fairseq2` and `seamless_communication`**

`fairseq2` must be installed from the [official wheel index](https://github.com/facebookresearch/fairseq2#installing-fairseq2) matching `torch==2.2.2+cu121` — do not `pip install fairseq2` directly. Then:
```bash
git clone https://github.com/facebookresearch/seamless_communication.git
python -m pip install -e seamless_communication/
```

**4. Verify**
```bash
python - <<'PY'
import torch, fairseq2, seamless_communication
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
PY
```
Should print `CUDA available: True` on a GPU machine, with no import errors.

---

## Directory Layout

```
GDPS_4lang/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_pipeline.sh                       # ★ one-shot entry point: data processing → training → gradient analysis → grouped fine-tuning → inference → eval
│   │
│   ├── finetune_multilang.py                 # Stage 1: Unified Fine-Tuning (UFT) trainer
│   ├── finetune_multilang_main.py
│   ├── finetune_B1_trainer.py                # Stage 2: grouped fine-tuning trainer
│   │
│   ├── gradient_conflict_analysis.py         # Gradient analysis: language grouping + which layer to specialize
│   ├── gradient_conflict_analysis_b1.py
│   ├── gradient_analysis_advanced.py         # Gradient analysis: energy allocation / share_ratio computation
│   ├── gradient_analysis_advanced_b1.py
│   ├── generate_config_from_gradient_analysis.py
│   │
│   ├── dataloader_with_fbank.py              # Data loader
│   ├── generate_fbank_features.py            # Audio -> fbank features
│   ├── prepare_manifest.py                   # TSV -> manifest
│   ├── rebalance_manifest.py                 # Rebalance per-language sample counts
│   ├── merge_manifests.py                    # Merge per-language manifests
│   ├── merge_multilang_models.py
│   ├── count_seamlessm4t_params.py
│   ├── run_ggmf_pipeline.py
│   │
│   ├── evaluate_multilang.py
│   └── evaluate_multilang_s2t.py             # BLEU / chrF++ / COMET evaluation
│
├── models/seamless_m4t_medium/               # Grouped-sharing model implementation
│   ├── __init__.py
│   ├── b1_minimal.py                         # Core: language grouping + layer decomposition
│   ├── dataset.py
│   ├── modeling_seamless_m4t_medium.py
│   ├── modeling_seamless_m4t_medium_B1.py
│   └── train.py
│
└── cli/m4t/predict/                          # Inference entry points
    ├── __init__.py
    ├── predict.py                            # Zero-shot / UFT inference
    ├── predict-using_4lang.py
    └── predict_b1_translator-4lang.py        # Grouped-model inference
```

---

## How to Run

### Data Prerequisites

You need the raw per-language TSVs (`{split}_{lang}_eng.tsv`), with a column pointing at the original `.wav` audio files. The source corpora are public: IWSLT22 (Tunisian), BIG-C (Bemba), LoResMT (Estonian), IWSLT23 (Irish).

### One-shot Run

```bash
bash scripts/run_pipeline.sh
```

Edit the `USER CONFIG` block at the top of the script first (data path, output path); nothing else needs to change. The script runs, in order:

1. **Data processing**: raw audio → fbank features → per-language manifests → merge/rebalance
2. **Unified Fine-Tuning (UFT)**: joint fine-tuning on the 4 merged languages, producing the base checkpoint
3. **Gradient analysis**: computes cross-language gradient similarity on the UFT checkpoint to determine the language grouping and the sharing ratio (`share_ratio`)
4. **Grouped fine-tuning**: decomposes one layer into a "shared + per-group" structure based on the analysis, then continues training
5. **Inference**: runs inference on the test set for each of the 4 languages
6. **Evaluation**: outputs BLEU / TER / chrF / chrF++ / BERTScore / COMET

Each `.py` script can also be invoked standalone (run with `--help` to see its arguments).

---


## Notes

- Most of the code here is reorganized from the `facebookresearch/seamless_communication` repository into a cleaner directory layout; `run_pipeline.sh` and `requirements.txt` were newly written for this repo to provide a clean, directly runnable entry point.
- The paper also reports several ablation studies (specializing different layers, different sharing ratios, a LoRA baseline, etc.); the corresponding code is not included yet.
- The trainer scripts (`finetune_multilang.py`, `finetune_B1_trainer.py`) rely on the project's relative directory layout to import the model implementation (`models/seamless_m4t_medium/`) — run `run_pipeline.sh` from the project root and no further setup is needed.
