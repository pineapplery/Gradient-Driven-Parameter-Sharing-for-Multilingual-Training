#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
B1架构模型推理脚本 - 基于fairseq2 Translator API

基于原始predict.py，修改为支持加载B1架构的checkpoint。

关键修改：
1. 在加载模型后、加载checkpoint前，先应用B1架构
2. 支持指定B1架构参数（share_ratio, target_module, num_groups）
3. 支持2-group和4-group分组方案
4. 支持多种目标模块选择（encoder_ffn1, encoder_ffn2, adapter_ffn, intermediate_ffn）
5. 其余推理流程与原始predict.py保持一致

B1架构支持的分组方案：
  - 2-group (默认): bem vs others (各1024维)
  - 4-group: aeb, bem, est, gle分别独立 (各512维)

用法示例：
  # 单个音频文件 - 2-group模式，encoder_ffn2（原始）
  python predict_b1_translator.py audio.wav \
    --task S2TT --tgt_lang eng --src_lang aeb \
    --load_checkpoint /path/to/checkpoint_2group.pt \
    --share_ratio 0.5 --num_groups 2

  # 单个音频文件 - 2-group模式，encoder_ffn1（新增）
  python predict_b1_translator.py audio.wav \
    --task S2TT --tgt_lang eng --src_lang bem \
    --load_checkpoint /path/to/checkpoint_ffn1.pt \
    --share_ratio 0.5 --num_groups 2 \
    --target_module encoder_ffn1

  # 单个音频文件 - 4-group模式
  python predict_b1_translator.py audio.wav \
    --task S2TT --tgt_lang eng --src_lang bem \
    --load_checkpoint /path/to/checkpoint_4group.pt \
    --share_ratio 0.5 --num_groups 4

  # TSV批量推理 - adapter_ffn模式
  python predict_b1_translator.py test.tsv \
    --task S2TT --tgt_lang eng \
    --load_checkpoint /path/to/adapter_checkpoint.pt \
    --share_ratio 0.5 --num_groups 2 \
    --target_module adapter_ffn \
    --output_path results.tsv
"""

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from fairseq2.generation import NGramRepeatBlockProcessor

from seamless_communication.inference import SequenceGeneratorOptions, Translator

# 导入B1架构
# 文件位置: /mnt/inspurfs/user-fs/224040284/workspace/seamless_communication/src/seamless_communication/cli/m4t/predict/predict_b1_translator.py
# B1模块位置: /mnt/inspurfs/user-fs/224040284/workspace/seamless_communication/src/seamless_communication/models/seamless_m4t_medium/b1_minimal.py
# 从 cli/m4t/predict/ 到 models/seamless_m4t_medium/: parent.parent.parent / "models" / "seamless_m4t_medium"
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "seamless_m4t_medium"))
from b1_minimal import apply_b1_to_model, set_current_group, get_group_for_lang

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["ASR", "S2ST", "S2TT"],
        help=(
            "* `ASR` -- automatic speech recognition (transcription);"
            "* `S2ST` -- speech to speech translation;"
            "* `S2TT` -- speech to text translation;"
        )
    )
    parser.add_argument(
        "--tgt_lang", type=str, help="Target language to translate/transcribe into."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language (for B1 group routing).",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the generated audio or TSV results.",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Base model name (`seamlessM4T_medium`, "
            "`seamlessM4T_large`, `seamlessM4T_v2_large`)"
        ),
        default="seamlessM4T_medium",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        help="Path to B1 checkpoint .pt file (REQUIRED for B1 inference).",
        required=True,
    )
    parser.add_argument(
        "--share_ratio",
        type=float,
        help="Share ratio for shared vs group parameters (default: 0.5). "
             "Controls k_shared = int(1024 * share_ratio) and "
             "r_shared = int(4096 * share_ratio). "
             "Example: 0.5 → r_shared=2048, k_shared=512",
        default=0.5,
    )
    parser.add_argument(
        "--target_module",
        type=str,
        help="Target module to apply B1 architecture (default: encoder_ffn2). "
             "Options: encoder_ffn1, encoder_ffn2, encoder_layer10_ffn2, adapter_ffn, intermediate_ffn",
        choices=['encoder_ffn1', 'encoder_ffn2', 'encoder_layer10_ffn2', 'adapter_ffn', 'intermediate_ffn'],
        default='encoder_ffn2',
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        help="Number of language groups in B1 architecture (default: 2). "
             "2 = bem vs others (1024 each); "
             "4 = aeb/bem/est/gle separate (512 each). "
             "Both use 2048 shared parameters.",
        choices=[2, 4],
        default=2,
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        help="Initialization strategy for group FFN2 parameters (default: residual). "
             "'residual': based on checkpoint residuals; "
             "'random': Kaiming initialization.",
        choices=["residual", "random"],
        default="residual",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        help="Dropout rate for FFN2 (shared and group) (default: 0.0, disabled)",
        default=0.0,
    )
    parser.add_argument(
        "--use_noise_init",
        action="store_true",
        help="Use small noise initialization for expanded dimensions (default: False, use zero padding)",
        default=False,
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        help="Vocoder model name",
        default="vocoder_v2",
    )
    # Text generation args.
    parser.add_argument(
        "--text_generation_beam_size",
        type=int,
        help="Beam size for incremental text decoding.",
        default=5,
    )
    parser.add_argument(
        "--text_generation_max_len_a",
        type=int,
        help="`a` in `ax + b` for incremental text decoding.",
        default=1,
    )
    parser.add_argument(
        "--text_generation_max_len_b",
        type=int,
        help="`b` in `ax + b` for incremental text decoding.",
        default=200,
    )
    parser.add_argument(
        "--text_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental text decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        help="Size of ngram repeat block for both text & unit decoding.",
        default=4,
    )
    # Unit generation args.
    parser.add_argument(
        "--unit_generation_beam_size",
        type=int,
        help=(
            "Beam size for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=5,
    )
    parser.add_argument(
        "--unit_generation_max_len_a",
        type=int,
        help=(
            "`a` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=25,
    )
    parser.add_argument(
        "--unit_generation_max_len_b",
        type=int,
        help=(
            "`b` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=50,
    )
    parser.add_argument(
        "--unit_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental unit decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--unit_generation_ngram_filtering",
        type=bool,
        help=(
            "If True, removes consecutive repeated ngrams"
            "from the decoded unit output."
        ),
        default=False,
    )
    parser.add_argument(
        "--text_unk_blocking",
        type=bool,
        help=(
            "If True, set penalty of UNK to inf in text generator "
            "to block unk output."
        ),
        default=False,
    )
    return parser


def set_generation_opts(
    args: Namespace,
) -> Tuple[SequenceGeneratorOptions, SequenceGeneratorOptions]:
    # Set text, unit generation opts.
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=args.text_generation_beam_size,
        soft_max_seq_len=(
            args.text_generation_max_len_a,
            args.text_generation_max_len_b,
        ),
    )
    if args.text_unk_blocking:
        text_generation_opts.unk_penalty = torch.inf
    if args.text_generation_ngram_blocking:
        text_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )

    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=args.unit_generation_beam_size,
        soft_max_seq_len=(
            args.unit_generation_max_len_a,
            args.unit_generation_max_len_b,
        ),
    )
    if args.unit_generation_ngram_blocking:
        unit_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )
    return text_generation_opts, unit_generation_opts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1 model inference on supported tasks using Translator."
    )
    parser.add_argument("input", type=str, help="Audio WAV file path, TSV file path, or text input.")

    parser = add_inference_arguments(parser)
    args = parser.parse_args()
    if not args.task or not args.tgt_lang:
        raise Exception(
            "Please provide required arguments for evaluation -  task, tgt_lang"
        )

    if args.task.upper() in {"S2ST", "T2ST"} and args.output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running B1 inference on {device=} with {dtype=}.")
    logger.info(f"B1 architecture configuration:")
    logger.info(f"  - target_module: {args.target_module}")
    logger.info(f"  - num_groups: {args.num_groups}")
    logger.info(f"  - share_ratio: {args.share_ratio}")
    r_shared = int(4096 * args.share_ratio)
    r_remaining = 4096 - r_shared
    logger.info(f"  - r_shared: {r_shared}")
    if args.num_groups == 2:
        r_group = r_remaining // 2
        logger.info(f"  - r_group: {r_group} (bem, others)")
    else:
        r_group_4 = r_remaining // 4
        logger.info(f"  - r_group: {r_group_4} each (aeb, bem, est, gle)")
    logger.info(f"  - init_strategy: {args.init_strategy}")
    logger.info(f"  - dropout_rate: {args.dropout_rate}")
    logger.info(f"  - use_noise_init: {args.use_noise_init}")

    # 1. 创建translator（加载基础模型）
    logger.info(f"\n=== Loading Base Model: {args.model_name} ===")
    translator = Translator(args.model_name, args.vocoder_name, device, dtype=dtype)

    # 2. 应用B1架构（在加载checkpoint前！）
    logger.info("\n=== Applying B1 Architecture to Model ===")
    translator.model = apply_b1_to_model(
        translator.model,
        share_ratio=args.share_ratio,
        dropout_rate=args.dropout_rate,
        init_strategy=args.init_strategy,
        num_groups=args.num_groups,
        target_module=args.target_module,
        use_noise_init=args.use_noise_init,
    )

    # 3. 加载B1 checkpoint
    ckpt_path = args.load_checkpoint
    logger.info(f"\n=== Loading B1 Checkpoint ===")
    logger.info(f"Checkpoint path: {ckpt_path}")

    def _strip_prefix_if_present(state_dict, prefix="model."):
        """移除state_dict键的前缀"""
        keys = list(state_dict.keys())
        if any(k.startswith(prefix) for k in keys):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
        return state_dict

    try:
        raw = torch.load(ckpt_path, map_location="cpu")
        
        # 尝试不同的checkpoint格式
        if isinstance(raw, dict) and "model" in raw:
            state = raw["model"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            state = raw
        
        # 移除可能的前缀
        state = _strip_prefix_if_present(state, prefix="model.")
        
        # 加载到模型
        load_res = translator.model.load_state_dict(state, strict=False)
        logger.info(
            f"✓ B1 checkpoint loaded successfully!"
        )
        logger.info(
            f"  Missing keys: {len(load_res.missing_keys)}"
        )
        logger.info(
            f"  Unexpected keys: {len(load_res.unexpected_keys)}"
        )
        
        if load_res.missing_keys:
            logger.warning(f"  Missing keys (first 10): {load_res.missing_keys[:10]}")
        if load_res.unexpected_keys:
            logger.warning(f"  Unexpected keys (first 10): {load_res.unexpected_keys[:10]}")
            
    except Exception as e:
        logger.exception(f"Failed to load B1 checkpoint {ckpt_path}: {e}")
        raise
    
    # 记录加载完毕的配置信息
    logger.info("\n=== B1 Model Ready for Inference ===")
    logger.info(f"Target module: {args.target_module}")
    if args.num_groups == 2:
        logger.info(f"Loaded 2-group B1 checkpoint (bem vs others)")
    else:
        logger.info(f"Loaded 4-group B1 checkpoint (aeb, bem, est, gle)")
    logger.info(f"Source language: {args.src_lang if args.src_lang else 'auto'}")
    logger.info(f"Target language: {args.tgt_lang}")
    logger.info(f"Share ratio: {args.share_ratio}")
    logger.info(f"Dropout rate: {args.dropout_rate}")
    logger.info(f"Noise initialization: {args.use_noise_init}")

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"\n=== Generation Options ===")
    logger.info(f"text_generation_opts: beam_size={args.text_generation_beam_size}, max_len=({args.text_generation_max_len_a}, {args.text_generation_max_len_b})")
    logger.info(f"unit_generation_opts: beam_size={args.unit_generation_beam_size}, max_len=({args.unit_generation_max_len_a}, {args.unit_generation_max_len_b})")
    logger.info(f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}")

    # If the input is a TSV file, extract audio paths
    if args.input.endswith(".tsv"):
        logger.info("\n=== TSV Batch Inference Mode ===")
        logger.info(f"Input TSV: {args.input}")
        tsv_data = pd.read_csv(args.input, sep="\t")
        
        # 检查必需的列
        if 'audio' not in tsv_data.columns:
            raise ValueError("TSV file must contain 'audio' column")
        
        audio_paths = tsv_data['audio'].tolist()
        logger.info(f"Extracted {len(audio_paths)} audio paths from TSV file.")

        # 准备保存结果的列表
        results = []

        for idx, audio_path in enumerate(audio_paths):
            logger.info(f"\n--- Processing [{idx+1}/{len(audio_paths)}]: {audio_path} ---")
            
            # 设置语言组（从src_lang推断）
            if args.src_lang:
                group = get_group_for_lang(args.src_lang)
                set_current_group(group)
                logger.info(f"Language group: {group} (src_lang: {args.src_lang})")

            try:
                if audio_path.endswith(".npy"):
                    # 加载.npy fbank特征文件
                    logger.info(f"Loading .npy fbank features: {audio_path}")
                    wav = np.load(audio_path)  # shape: (time, 80)
                    
                    if wav.ndim == 1:
                        wav = np.expand_dims(wav, axis=0)
                    
                    wav = torch.tensor(wav, dtype=torch.float32)
                    
                    # 转换为模型的dtype
                    if dtype == torch.float16:
                        wav = wav.half()
                    
                    # 检查序列长度
                    seq_len = wav.shape[0]
                    min_kernel_size = 31
                    if seq_len < min_kernel_size:
                        logger.warning(f"[SKIP] Audio too short (seq_len={seq_len} < {min_kernel_size}), skipping.")
                        results.append({
                            'audio': audio_path,
                            'text': '[ERROR: Audio too short]'
                        })
                        continue
                    
                    sample_rate = 16_000
                else:
                    # 加载普通音频文件
                    logger.info(f"Loading audio file: {audio_path}")
                    wav, sample_rate = torchaudio.load(audio_path)

                # Ensure the tensor has the correct shape (channels, samples)
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)

                translator_input = torchaudio.functional.resample(
                    wav, orig_freq=sample_rate, new_freq=16_000
                )

                # 推理
                text_output, speech_output = translator.predict(
                    translator_input,
                    args.task,
                    args.tgt_lang,
                    src_lang=args.src_lang,
                    text_generation_opts=text_generation_opts,
                    unit_generation_opts=unit_generation_opts,
                    unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
                )

                # 保存翻译结果
                translated_text = str(text_output[0]) if text_output else ''
                results.append({
                    'audio': audio_path,
                    'text': translated_text
                })

                logger.info(f"✓ Translated text: {translated_text}")

                if speech_output is not None and args.output_path:
                    output_audio_path = Path(args.output_path) / f"{Path(audio_path).stem}.wav"
                    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(
                        str(output_audio_path),
                        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                        sample_rate=speech_output.sample_rate,
                    )
                    logger.info(f"✓ Saved translated audio to: {output_audio_path}")
                    
            except Exception as e:
                logger.exception(f"Error processing {audio_path}: {e}")
                results.append({
                    'audio': audio_path,
                    'text': f'[ERROR: {str(e)}]'
                })

        # 保存所有结果到TSV文件
        if results and args.output_path:
            output_tsv = args.output_path if str(args.output_path).endswith('.tsv') else Path(args.output_path) / "results.tsv"
            logger.info(f"\n=== Saving Results ===")
            logger.info(f"Output TSV: {output_tsv}")
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_tsv, sep="\t", index=False)
            logger.info(f"✓ Saved {len(results)} results to {output_tsv}")
    else:
        # Single audio/text file mode
        logger.info("\n=== Single File Inference Mode ===")
        
        # 设置语言组
        if args.src_lang:
            group = get_group_for_lang(args.src_lang)
            set_current_group(group)
            logger.info(f"Language group: {group} (src_lang: {args.src_lang})")

        if args.task.upper() in {"S2ST", "ASR", "S2TT"}:
            wav, sample_rate = torchaudio.load(args.input)
            translator_input = torchaudio.functional.resample(
                wav, orig_freq=sample_rate, new_freq=16_000
            )
        else:
            translator_input = args.input

        text_output, speech_output = translator.predict(
            translator_input,
            args.task,
            args.tgt_lang,
            src_lang=args.src_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
        )

        if speech_output is not None and args.output_path:
            logger.info(f"Saving translated audio in {args.tgt_lang}")
            torchaudio.save(
                str(args.output_path),
                speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                sample_rate=speech_output.sample_rate,
            )
            logger.info(f"✓ Saved to: {args.output_path}")
            
        logger.info(f"\n=== Translation Result ===")
        logger.info(f"Translated text in {args.tgt_lang}: {text_output[0]}")


if __name__ == "__main__":
    main()
