# GDPS for Interspeech 2026

本目录包含 **Interspeech 2026 论文**《Automated Gradient-Driven Parameter Sharing for Low-Resource Multilingual Speech-to-Text Translation》的代码，支持 4 种语言的语音翻译：**aeb, bem, est, gle → eng**。

骨干模型为 **`seamlessM4T_medium`**。GDPS（Gradient-Driven Parameter Sharing）的核心思路是：先在多语言数据上统一微调（UFT），再基于梯度统计信息，把编码器中的一个瓶颈层分解为"共享分支 + 语言分组专属分支"，让相近的语言共享参数、差异较大的语言各自训练专属参数。

---

## 环境配置

开发与验证环境为 **Python 3.10 + CUDA 12.1 + PyTorch 2.2.2**。具体依赖版本见 [`requirements.txt`](requirements.txt)。

**1. 创建环境**
```bash
conda create -n gdps python=3.10 -y
conda activate gdps
```

**2. 安装 Python 依赖**（会通过文件中声明的 extra index URL 拉取 CUDA 12.1 版本的 `torch`/`torchaudio`）
```bash
python -m pip install -r requirements.txt
```

**3. 安装 `fairseq2` 与 `seamless_communication`**

`fairseq2` 需要从匹配 `torch==2.2.2+cu121` 的[官方 wheel 索引](https://github.com/facebookresearch/fairseq2#installing-fairseq2)安装，不能直接 `pip install fairseq2`。然后：
```bash
git clone https://github.com/facebookresearch/seamless_communication.git
python -m pip install -e seamless_communication/
```

**4. 验证**
```bash
python - <<'PY'
import torch, fairseq2, seamless_communication
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
PY
```
应打印 `CUDA available: True`（GPU 机器上）且无报错。

---

## 目录结构

```
GDPS_4lang/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_pipeline.sh                       # ★ 一键运行入口：数据处理 → 训练 → 梯度分析 → 分组微调 → 推理 → 评估
│   │
│   ├── finetune_multilang.py                 # 阶段1：统一微调（UFT）训练器
│   ├── finetune_multilang_main.py
│   ├── finetune_B1_trainer.py                # 阶段2：分组微调训练器
│   │
│   ├── gradient_conflict_analysis.py         # 梯度分析：语言分组 + 判断特化哪一层
│   ├── gradient_conflict_analysis_b1.py
│   ├── gradient_analysis_advanced.py         # 梯度分析：能量分配 / share_ratio 计算
│   ├── gradient_analysis_advanced_b1.py
│   ├── generate_config_from_gradient_analysis.py
│   │
│   ├── dataloader_with_fbank.py              # 数据加载器
│   ├── generate_fbank_features.py            # 音频 → fbank 特征
│   ├── prepare_manifest.py                   # TSV → manifest
│   ├── rebalance_manifest.py                 # 各语言样本重平衡
│   ├── merge_manifests.py                    # 合并多语言 manifest
│   ├── merge_multilang_models.py
│   ├── count_seamlessm4t_params.py
│   ├── run_ggmf_pipeline.py
│   │
│   ├── evaluate_multilang.py
│   └── evaluate_multilang_s2t.py             # BLEU / chrF++ / COMET 评估
│
├── models/seamless_m4t_medium/               # 分组共享模型的实现
│   ├── __init__.py
│   ├── b1_minimal.py                         # 核心：语言分组 + 层分解实现
│   ├── dataset.py
│   ├── modeling_seamless_m4t_medium.py
│   ├── modeling_seamless_m4t_medium_B1.py
│   └── train.py
│
└── cli/m4t/predict/                          # 推理入口
    ├── __init__.py
    ├── predict.py                            # 零样本 / UFT 推理
    ├── predict-using_4lang.py
    └── predict_b1_translator-4lang.py        # 分组模型推理
```

---

## 如何运行

### 前提数据

需要准备好 4 种语言各自的原始 TSV 文件（`{split}_{lang}_eng.tsv`），列中包含指向原始 `.wav` 音频的路径。语料来自公开数据集：IWSLT22（Tunisian）、BIG-C（Bemba）、LoResMT（Estonian）、IWSLT23（Irish）。

### 一键运行

```bash
bash scripts/run_pipeline.sh
```

运行前先编辑脚本顶部的 `USER CONFIG` 块（数据路径、输出路径），其余无需改动。脚本会按顺序完成：

1. **数据处理**：原始音频 → fbank 特征 → 各语言 manifest → 合并/重平衡
2. **统一微调（UFT）**：在四语言合并数据上联合微调，得到基础 checkpoint
3. **梯度分析**：在 UFT checkpoint 上计算语言间梯度相似度，得到语言分组和共享比例（`share_ratio`）
4. **分组微调**：基于分析结果，把模型中的一层分解为"共享 + 分组专属"结构并继续训练
5. **推理**：对 4 种语言的测试集分别推理
6. **评估**：输出 BLEU / TER / chrF / chrF++ / BERTScore / COMET

也可以单独调用其中某个 `.py` 脚本（各脚本均通过 `--help` 查看参数说明）。

---


## 说明

- 本目录中的代码大多是从`facebookresearch/seamless_communication`代码仓库整理而来，目录结构做了重新组织；`run_pipeline.sh`、`requirements.txt` 为本仓库新写，用于提供一个干净、可直接运行的入口。
- 论文中还包含若干消融实验（不同层特化、不同共享比例、LoRA baseline 等），相关代码暂未包含。
- 训练器脚本（`finetune_multilang.py`、`finetune_B1_trainer.py`）依赖项目相对目录结构来导入模型实现（`models/seamless_m4t_medium/`），请在项目根目录下运行 `run_pipeline.sh`，无需额外配置。
