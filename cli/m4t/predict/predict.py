# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # Add this import for handling .npy files
import pandas as pd  # Add this import for handling TSV files
import torch
import torchaudio
from fairseq2.generation import NGramRepeatBlockProcessor

from seamless_communication.inference import SequenceGeneratorOptions, Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
# Translator emits per-sample INFO logs; default to WARNING for large-batch TSV throughput.
logging.getLogger("seamless_communication.inference.translator").setLevel(logging.WARNING)


def normalize_model_name(model_name: str) -> str:
    """Normalize HF aliases to unity model names expected by Translator."""
    name = (model_name or "").strip()
    lowered = name.lower()
    if lowered in {
        "facebook/hf-seamless-m4t-large",
        "hf-seamless-m4t-large",
        "seamlessm4t_large",
        "seamlessm4t-large",
    }:
        return "seamlessM4T_large"
    if lowered in {
        "facebook/hf-seamless-m4t-medium",
        "hf-seamless-m4t-medium",
        "seamlessm4t_medium",
        "seamlessm4t-medium",
    }:
        return "seamlessM4T_medium"
    return model_name


def _expected_checkpoint_filename(model_name: str) -> Optional[str]:
    mapping = {
        "seamlessM4T_v2_large": "seamlessM4T_v2_large.pt",
        "seamlessM4T_large": "multitask_unity_large.pt",
        "seamlessM4T_medium": "multitask_unity_medium.pt",
    }
    return mapping.get(model_name)


def configure_cache_dirs(cache_root: Optional[str], effective_model_name: Optional[str] = None) -> None:
    """Configure cache directories with backward-compatible layout detection.

    Supports both layouts:
    1) <cache_root>/fairseq2/assets/... (preferred)
    2) <cache_root>/assets/... (legacy/flat)
    """
    if not cache_root:
        return

    root = Path(cache_root)
    nested_hf = root / "hf"
    nested_fairseq2 = root / "fairseq2"

    # Keep compatibility with existing caches while defaulting to nested layout.
    if (nested_fairseq2 / "assets").exists():
        fairseq2_cache = nested_fairseq2
        fairseq2_layout = "nested"
    elif (root / "assets").exists():
        fairseq2_cache = root
        fairseq2_layout = "flat"
    else:
        fairseq2_cache = nested_fairseq2
        fairseq2_layout = "nested(new)"

    if (root / "hub").exists() and not nested_hf.exists():
        hf_cache = root
        hf_layout = "flat"
    else:
        hf_cache = nested_hf
        hf_layout = "nested"

    hf_cache.mkdir(parents=True, exist_ok=True)
    fairseq2_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["FAIRSEQ2_CACHE_DIR"] = str(fairseq2_cache)

    logger.info("\n=== Setting Cache Directory ===")
    logger.info(f"Cache root: {root}")
    logger.info(f"✓ Set HF_HOME={hf_cache} (layout={hf_layout})")
    logger.info(f"✓ Set FAIRSEQ2_CACHE_DIR={fairseq2_cache} (layout={fairseq2_layout})")

    if effective_model_name:
        expected_ckpt = _expected_checkpoint_filename(effective_model_name)
        if expected_ckpt:
            hits = sorted((fairseq2_cache / "assets").glob(f"*/{expected_ckpt}"))
            if hits:
                logger.info(f"✓ Found local fairseq2 checkpoint: {hits[0]}")
            else:
                logger.warning(
                    "No local fairseq2 checkpoint found under "
                    f"{fairseq2_cache / 'assets'} for {expected_ckpt}; download may occur."
                )


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
        help="Source language, only required if input is text.",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the generated audio.",
        default=None,
    )
    parser.add_argument(
        "--batch_manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest for multi-language TSV inference in one process. "
            "Each item must include src_lang, input_tsv, output_tsv."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Base model name (`seamlessM4T_medium`, "
            "`seamlessM4T_large`, `seamlessM4T_v2_large`)"
        ),
        default="seamlessM4T_v2_large",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache dir. If set, export HF_HOME and FAIRSEQ2_CACHE_DIR before model load.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=200,
        help="Progress bar refresh interval in TSV mode. 0 means refresh every sample.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap for TSV rows in smoke tests. 0 means use all rows.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        help="Path to a local .pt checkpoint to load into the model (optional).",
        default=None,
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

#新增
def clean_text(text: str) -> str:
    """Clean and normalize predicted text for evaluation."""
    import re
    text = str(text).strip()
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    # Fix spacing between sentences
    text = re.sub(r'([.!?,;:])\s*([A-Z])', r'\1 \2', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def should_refresh_progress(idx: int, total: int, interval: int) -> bool:
    step = interval if interval > 0 else 1
    if idx == 0 or idx == total - 1:
        return True
    return ((idx + 1) % step) == 0


def render_progress(prefix: str, current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return

    ratio = max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    bar = ("#" * filled) + ("-" * (width - filled))
    print(
        f"\r[{prefix}] {current}/{total} {ratio * 100:6.2f}% |{bar}|",
        end="",
        flush=True,
    )
    if current >= total:
        print("", flush=True)


def load_batch_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        raise ValueError("--batch_manifest must contain a JSON list or an object with 'items'.")

    if not isinstance(items, list) or not items:
        raise ValueError("--batch_manifest has no items.")

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid batch item at index {idx}: expected object.")
        src_lang = item.get("src_lang")
        input_tsv = item.get("input_tsv") or item.get("input")
        output_tsv = item.get("output_tsv") or item.get("output") or item.get("output_path")
        if not src_lang or not input_tsv or not output_tsv:
            raise ValueError(
                f"Invalid batch item at index {idx}: require src_lang/input_tsv/output_tsv."
            )
        normalized.append(
            {
                "src_lang": str(src_lang),
                "input_tsv": str(input_tsv),
                "output_tsv": str(output_tsv),
            }
        )
    return normalized


def run_tsv_inference(
    translator: Translator,
    args: Namespace,
    input_tsv: str,
    output_tsv: Path,
    src_lang: Optional[str],
    text_generation_opts: SequenceGeneratorOptions,
    unit_generation_opts: SequenceGeneratorOptions,
) -> None:
    logger.info(f"Detected TSV file input for {src_lang}->{args.tgt_lang}. Extracting audio paths.")
    tsv_data = pd.read_csv(input_tsv, sep="\t")
    if "audio" not in tsv_data.columns:
        raise ValueError(f"TSV must contain an 'audio' column: {input_tsv}")

    if args.max_samples > 0 and len(tsv_data) > args.max_samples:
        logger.info(f"[SMOKE] Limiting {src_lang}->{args.tgt_lang} to first {args.max_samples} samples.")
        tsv_data = tsv_data.head(args.max_samples)

    audio_paths = tsv_data["audio"].tolist()
    logger.info(f"Extracted {len(audio_paths)} audio paths from TSV file.")

    results = []
    success_count = 0
    skipped_count = 0
    error_count = 0

    total_samples = len(audio_paths)
    progress_prefix = f"{src_lang}->{args.tgt_lang}"

    for idx, audio_path in enumerate(audio_paths):
        try:
            if audio_path.endswith(".npy"):
                npy = np.load(audio_path)
                if npy.ndim == 1:
                    npy = np.expand_dims(npy, axis=0)
                elif npy.ndim == 2 and npy.shape[0] == 80 and npy.shape[1] != 80:
                    # Some fbank files are stored as [80, T]; normalize to [T, 80].
                    npy = npy.T
                elif npy.ndim != 2:
                    raise ValueError(f"Unsupported npy rank={npy.ndim} for {audio_path}")

                wav = torch.tensor(npy, dtype=torch.float32)

                is_fbank = wav.ndim == 2 and wav.shape[1] == 80
                if is_fbank:
                    wav = wav.half()
                    seq_len = wav.shape[0]
                    min_kernel_size = 31
                    if seq_len < min_kernel_size:
                        logger.warning(
                            f"[SKIP] {audio_path} too short for model "
                            f"(seq_len={seq_len} < kernel_size={min_kernel_size}), skipping."
                        )
                        skipped_count += 1
                        continue

                sample_rate = 16_000
            else:
                wav, sample_rate = torchaudio.load(audio_path)

            if wav.ndim == 1:
                wav = wav.unsqueeze(0)

            if sample_rate != 16_000:
                translator_input = torchaudio.functional.resample(
                    wav, orig_freq=sample_rate, new_freq=16_000
                )
            else:
                translator_input = wav

            text_output, speech_output = translator.predict(
                translator_input,
                args.task,
                args.tgt_lang,
                src_lang=src_lang,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            )

            cleaned_text = clean_text(str(text_output[0])) if text_output else ""
            results.append({"audio": audio_path, "text": cleaned_text})
            success_count += 1

            if speech_output is not None:
                audio_dir = output_tsv.parent if output_tsv.suffix else output_tsv
                audio_dir.mkdir(parents=True, exist_ok=True)
                output_audio_path = audio_dir / Path(audio_path).stem
                torchaudio.save(
                    f"{output_audio_path}.wav",
                    speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                    sample_rate=speech_output.sample_rate,
                )
        except Exception as ex:
            error_count += 1
            logger.warning(f"[SKIP] [{idx+1}/{total_samples}] Failed on sample {audio_path}: {ex}")
            continue
        finally:
            if should_refresh_progress(idx, total_samples, args.log_interval):
                render_progress(progress_prefix, idx + 1, total_samples)

    if results:
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(results)} results to {output_tsv}")
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_tsv, sep="\t", index=False)

    logger.info(
        f"Completed {progress_prefix}: total={total_samples}, "
        f"success={success_count}, skipped={skipped_count}, errors={error_count}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4T inference on supported tasks using Translator. Supports 8 languages (aeb, bem, ckb, est, gle, hau, ibo, yor)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default=None,
        help="Audio WAV file path, TSV file path, or text input.",
    )

    parser = add_inference_arguments(parser)
    args = parser.parse_args()

    if not args.input and not args.batch_manifest:
        raise ValueError("Provide either positional input or --batch_manifest.")
    if args.batch_manifest and args.input:
        logger.info("Both input and --batch_manifest were provided; batch manifest mode takes precedence.")

    effective_model_name = normalize_model_name(args.model_name)

    configure_cache_dirs(args.cache_dir, effective_model_name)
    if not args.task or not args.tgt_lang:
        raise Exception(
            "Please provide required arguments for evaluation -  task, tgt_lang"
        )
    
    # Validate supported languages for multilingual checkpoint
    supported_langs = {"aeb", "bem", "ckb", "est", "gle", "hau", "ibo", "yor", "eng"}
    if args.src_lang and args.src_lang not in supported_langs:
        logger.warning(f"Source language '{args.src_lang}' may not be in the 8-language checkpoint. Supported: {supported_langs}")
    if args.tgt_lang not in supported_langs:
        logger.warning(f"Target language '{args.tgt_lang}' may not be in the 8-language checkpoint. Supported: {supported_langs}")

    if args.task.upper() in {"S2ST", "T2ST"} and args.output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running inference on {device=} with {dtype=}.")

    logger.info(f"Model (input): {args.model_name}")
    logger.info(f"Model (effective): {effective_model_name}")
    translator = Translator(effective_model_name, args.vocoder_name, device, dtype=dtype)

    # If a local checkpoint is provided, try to load it into the model.
    if getattr(args, 'load_checkpoint', None):
        ckpt_path = args.load_checkpoint
        logger.info(f"Loading checkpoint from {ckpt_path}")

        def _strip_prefix_if_present(state_dict, prefix="model."):
            # if keys start with prefix, strip it
            keys = list(state_dict.keys())
            if any(k.startswith(prefix) for k in keys):
                return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            return state_dict

        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw, dict) and "model" in raw:
                state = raw["model"]
            elif isinstance(raw, dict) and "model_state_dict" in raw:
                state = raw["model_state_dict"]
            else:
                state = raw
            state = _strip_prefix_if_present(state, prefix="model.")

            # Robust loading: filter out keys that don't exist or have shape mismatch.
            model_state = translator.model.state_dict()
            filtered_state = {}
            skipped_not_found = 0
            skipped_shape = 0
            for k, v in state.items():
                if k not in model_state:
                    skipped_not_found += 1
                    continue
                if model_state[k].shape != v.shape:
                    skipped_shape += 1
                    continue
                filtered_state[k] = v

            load_res = translator.model.load_state_dict(filtered_state, strict=False)
            logger.info(
                f"Checkpoint loaded. used={len(filtered_state)}, "
                f"skipped_not_found={skipped_not_found}, skipped_shape={skipped_shape}, "
                f"missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)}"
            )
        except Exception as e:
            logger.exception(f"Failed to load checkpoint {ckpt_path}: {e}")

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    if args.batch_manifest is not None:
        jobs = load_batch_manifest(args.batch_manifest)
        logger.info(f"Running batch manifest with {len(jobs)} language jobs in one process.")
        for job_idx, job in enumerate(jobs):
            src_lang = job["src_lang"]
            input_tsv = job["input_tsv"]
            output_tsv = Path(job["output_tsv"])
            logger.info(
                f"[JOB {job_idx + 1}/{len(jobs)}] "
                f"{src_lang}->{args.tgt_lang} | input={input_tsv} | output={output_tsv}"
            )
            run_tsv_inference(
                translator=translator,
                args=args,
                input_tsv=input_tsv,
                output_tsv=output_tsv,
                src_lang=src_lang,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
            )
        return

    # If the input is a TSV file, extract audio paths
    if args.input and args.input.endswith(".tsv"):
        if args.output_path is None:
            raise ValueError("output_path must be provided in TSV mode.")
        run_tsv_inference(
            translator=translator,
            args=args,
            input_tsv=args.input,
            output_tsv=Path(args.output_path),
            src_lang=args.src_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
        )
    else:
        # If the input is audio or text
        if args.input is None:
            raise ValueError("input is required for non-batch single-file mode.")
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

        if speech_output is not None:
            logger.info(f"Saving translated audio in {args.tgt_lang}")
            torchaudio.save(
                args.output_path,
                speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                sample_rate=speech_output.sample_rate,
            )
        logger.info(f"Translated text in {args.tgt_lang}: {text_output[0]}")


if __name__ == "__main__":
    main()
