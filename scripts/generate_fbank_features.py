
#!/usr/bin/env python3
"""
修正版fbank特征提取脚本：
根据调试结果，正确处理fbank_converter的输出格式
支持多语言和多split的批量处理
"""

import os
import logging
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='生成80维fbank特征文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
1. 单一语言模式（处理单一语言数据集，如CKB）：
   python generate_fbank_features.py \\
       --tsv_src /224040284/data/ckb_eng \\
       --fbank_out /224040284/data/ckb_eng/fbank_80

2. 多语言模式（处理多语言数据集，如African languages）：
   python generate_fbank_features.py \\
       --mode multilang \\
       --tsv_src /224040284/data/african_celtic_full_extracted \\
       --fbank_out /224040284/data/african_celtic_full_extracted/fbank_80

参数说明：
  --tsv_src: 源TSV文件所在路径（必需）
  
  --fbank_out: fbank文件输出基础路径（必需）
  
  --mode: 处理模式（默认：singlelang）
    - singlelang: 单一语言模式，输出路径为 fbank_out/{split}/
    - multilang: 多语言模式，输出路径为 fbank_out/{lang}/{split}/
  
  --gpu: GPU设备ID（默认为0，可选）
        """
    )
    
    parser.add_argument(
        '--tsv_src',
        type=str,
        required=True,
        help='源TSV文件所在路径'
    )
    
    parser.add_argument(
        '--fbank_out',
        type=str,
        required=True,
        help='fbank文件输出基础路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['singlelang', 'multilang'],
        default='singlelang',
        help='处理模式：singlelang为单一语言模式（默认），multilang为多语言模式'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU设备ID（默认为0）'
    )
    
    return parser.parse_args()


def initialize_paths(args):
    """根据命令行参数初始化路径"""
    tsv_src = Path(args.tsv_src)
    fbank_out = Path(args.fbank_out)
    
    if not tsv_src.exists():
        logger.error(f'TSV源路径不存在: {tsv_src}')
        sys.exit(1)
    
    return {
        'TSV_SRC_DIR': tsv_src,
        'FBANK_OUT_DIR': fbank_out,
        'mode': args.mode
    }


def get_lang_and_split_from_filename(filename):
    """
    从TSV文件名解析语言和split信息
    文件名格式：{split}_{lang}_eng.tsv
    例如：train_hau_eng.tsv → (train, hau)
    """
    stem = Path(filename).stem
    parts = stem.rsplit('_', 2)  # 从右边分割，分割次数为2
    
    if len(parts) == 3:
        split = parts[0]
        lang = parts[1]
        return split, lang
    else:
        return None, None


# 初始化
args = parse_arguments()
paths = initialize_paths(args)

TSV_SRC_DIR = paths['TSV_SRC_DIR']
FBANK_OUT_DIR = paths['FBANK_OUT_DIR']
MODE = paths['mode']

logger.info(f"运行模式: {MODE}")
logger.info(f"TSV源路径: {TSV_SRC_DIR}")
logger.info(f"FBANK输出路径: {FBANK_OUT_DIR}")

# 创建输出目录
FBANK_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置设备
try:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
except:
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

# 初始化官方使用的 audio_decoder 和 fbank_converter
audio_decoder = AudioDecoder(dtype=torch.float32, device=device)
fbank_converter = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=True,
    device=device,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
)

# 获取所有待处理 TSV 文件（排除已经处理的 npy TSV）
if MODE == 'singlelang':
    # 单一语言模式：查找当前路径下的 {split}_{lang}_eng.tsv 文件
    tsv_files = list(TSV_SRC_DIR.glob("*.tsv"))
    tsv_files = [f for f in tsv_files if not f.stem.endswith("_npy") and "_eng.tsv" in f.name]
else:  # multilang 模式
    # 多语言模式：递归查找所有 {split}_{lang}_eng.tsv 文件
    tsv_files = []
    for tsv_file in TSV_SRC_DIR.rglob("*.tsv"):
        if not tsv_file.stem.endswith("_npy") and "_eng.tsv" in tsv_file.name:
            tsv_files.append(tsv_file)

logger.info(f"Found {len(tsv_files)} source TSV files")

def extract_features_for_file(tsv_path: Path):
    """
    针对一个 TSV，逐条提取音频 fbank 特征
    """
    logger.info(f"Processing: {tsv_path.name}")
    df = pd.read_csv(tsv_path, sep="\t")

    if MODE == 'singlelang':
        # 单一语言模式：从文件名解析split和语言
        split_type, lang = get_lang_and_split_from_filename(tsv_path.name)
        if not split_type or not lang:
            logger.warning(f"Failed to parse TSV name format: {tsv_path.name}, skipping")
            return
        
        # 单一语言模式：输出路径为 FBANK_OUT_DIR / {split_type}
        output_subdir = FBANK_OUT_DIR / split_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        lang_pair = f"{lang}_eng"  # 用于输出TSV文件名
    else:  # multilang 模式
        # 多语言模式：从文件名解析split和语言
        split_type, lang = get_lang_and_split_from_filename(tsv_path.name)
        if not split_type or not lang:
            logger.warning(f"Failed to parse TSV name format: {tsv_path.name}, skipping")
            return
        
        # 多语言模式：输出路径为 FBANK_OUT_DIR / {lang} / {split_type}
        output_subdir = FBANK_OUT_DIR / lang / split_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        lang_pair = f"{lang}_eng"  # 用于输出TSV文件名

    new_rows = []
    success_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        audio_path = row.get("audio")
        if audio_path is None or (isinstance(audio_path, float) and np.isnan(audio_path)):
            logger.warning(f"Row {idx}: Missing audio path, skipping")
            error_count += 1
            continue
            
        if not os.path.exists(str(audio_path)):
            logger.warning(f"Audio not found: {audio_path}")
            error_count += 1
            continue

        try:
            # 读取文件 bytes
            with open(str(audio_path), "rb") as fb:
                block = MemoryBlock(fb.read())

            # 解码音频
            decoded_audio = audio_decoder(block)
            
            # 检查采样率，如果不是16000则进行重采样
            sample_rate = decoded_audio.get("sample_rate")
            waveform = decoded_audio["waveform"]
            
            if sample_rate is not None and sample_rate != 16000:
                logger.info(f"Resampling {Path(audio_path).name} from {sample_rate}Hz to 16000Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000,
                    dtype=torch.float32
                ).to(device)
                waveform = resampler(waveform)
                decoded_audio["waveform"] = waveform
                decoded_audio["sample_rate"] = 16000

            # 提取 fbank 特征
            with torch.no_grad():
                fbank_out = fbank_converter(decoded_audio)
                # 根据调试结果，fbank_out是一个字典，包含fbank键
                # fbank_out["fbank"]直接就是形状为(time, 80)的tensor
                fbank_tensor = fbank_out["fbank"]
                fbank_arr = fbank_tensor.cpu().numpy()

            # 规范检查
            if fbank_arr.size == 0:
                logger.error(f"Empty fbank array for {audio_path}")
                error_count += 1
                continue

            # 检查形状是否为(time, 80)
            if fbank_arr.shape[-1] != 80:
                logger.error(f"Invalid fbank shape {fbank_arr.shape} for {audio_path}, expected (time, 80)")
                error_count += 1
                continue

            # 保存为 npy
            audio_stem = Path(audio_path).stem
            out_npy = output_subdir / f"{audio_stem}.npy"
            np.save(str(out_npy), fbank_arr.astype(np.float32))

            # 记录结果，保留原始行数据，只更新audio列
            new_row = row.to_dict()
            new_row["audio"] = str(out_npy)
            new_rows.append(new_row)

            success_count += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx+1}/{len(df)} rows (success: {success_count}, error: {error_count})")

        except Exception as e:
            logger.error(f"Failed feature extract for {audio_path}: {e}")
            error_count += 1
            continue

    # 写出新的 npy TSV
    if new_rows:
        out_df = pd.DataFrame(new_rows)
        out_name = f"{split_type}_{lang_pair}_npy.tsv"
        
        # 所有模式：输出到与源TSV同级的目录
        out_path = tsv_path.parent / out_name
        
        out_df.to_csv(str(out_path), sep="\t", index=False)
        logger.info(f"Saved fbank TSV: {out_path}")
        logger.info(f"Summary: success={success_count}, error={error_count}")
    else:
        logger.warning(f"No valid rows extracted for {tsv_path.name}")

# 遍历所有 TSV
logger.info(f"\n{'='*80}")
logger.info("开始处理所有TSV文件...")
logger.info(f"{'='*80}\n")

for tsv_file in tsv_files:
    extract_features_for_file(tsv_file)

logger.info(f"\n{'='*80}")
logger.info("✓ 所有文件处理完成!")
logger.info(f"{'='*80}")

logger.info(f"\nfbank文件输出位置: {FBANK_OUT_DIR}")
logger.info(f"_npy.tsv文件位置: 与原始TSV文件同级目录")

# 统计输出的npy.tsv文件
if MODE == 'singlelang':
    npy_tsv_files = list(TSV_SRC_DIR.glob("*_npy.tsv"))
else:  # multilang
    npy_tsv_files = list(TSV_SRC_DIR.rglob("*_npy.tsv"))

logger.info(f"生成的_npy.tsv文件: {len(npy_tsv_files)} 个")
for npy_tsv in sorted(npy_tsv_files):
    logger.info(f"  - {npy_tsv}")





# #!/usr/bin/env python3
# """
# 修复版fbank特征提取脚本：
# 添加采样率转换功能，支持不同采样率的音频文件
# """

# import os
# import logging
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import torchaudio

# from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
# from fairseq2.memory import MemoryBlock

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s -- %(name)s: %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ----------- 配置区域 -----------
# BASE_DIR = Path("/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big")
# TSV_SRC_DIR = BASE_DIR / "file_80_original"
# FBANK_OUT_DIR = BASE_DIR / "fbank_features_with_resample"
# TSV_OUT_DIR = BASE_DIR
# TARGET_SAMPLE_RATE = 16000  # 目标采样率
# # ---------------------------------

# # 创建输出目录
# FBANK_OUT_DIR.mkdir(parents=True, exist_ok=True)

# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device: {device}")

# # 初始化官方使用的 audio_decoder 和 fbank_converter
# audio_decoder = AudioDecoder(dtype=torch.float32, device=device)
# fbank_converter = WaveformToFbankConverter(
#     num_mel_bins=80,
#     waveform_scale=2**15,
#     channel_last=True,
#     standardize=True,
#     device=device,
#     dtype=torch.float16 if device.type == "cuda" else torch.float32,
# )

# # 初始化重采样器
# resampler = torchaudio.transforms.Resample(
#     orig_freq=48000,  # 常见的高采样率
#     new_freq=TARGET_SAMPLE_RATE,
#     dtype=torch.float32
# ).to(device)

# # 获取所有待处理 TSV 文件（排除已经处理的 npy TSV）
# tsv_files = list(TSV_SRC_DIR.glob("*.tsv"))
# tsv_files = [f for f in tsv_files if not f.stem.endswith("_npy")]
# logger.info(f"Found {len(tsv_files)} source TSV files")

# def resample_audio(waveform, original_sample_rate, target_sample_rate):
#     """
#     重采样音频到目标采样率

#     Args:
#         waveform: 音频波形 tensor
#         original_sample_rate: 原始采样率
#         target_sample_rate: 目标采样率

#     Returns:
#         resampled_waveform: 重采样后的音频波形
#     """
#     if original_sample_rate == target_sample_rate:
#         return waveform

#     # 创建重采样器
#     resampler = torchaudio.transforms.Resample(
#         orig_freq=original_sample_rate,
#         new_freq=target_sample_rate,
#         dtype=torch.float32
#     ).to(waveform.device)

#     # 重采样
#     resampled_waveform = resampler(waveform)

#     return resampled_waveform

# def extract_features_for_file(tsv_path: Path):
#     """
#     针对一个 TSV，逐条提取音频 fbank 特征
#     """
#     logger.info(f"Processing: {tsv_path.name}")
#     df = pd.read_csv(tsv_path, sep="\t")

#     # 将输出放在按 split + langpair 命名的子目录
#     parts = tsv_path.stem.split("_", 1)
#     if len(parts) != 2:
#         logger.warning(f"Unexpected TSV name format: {tsv_path.name}, skipping")
#         return
#     split_type, lang_pair = parts
#     output_subdir = FBANK_OUT_DIR / f"{split_type}_{lang_pair}"
#     output_subdir.mkdir(parents=True, exist_ok=True)

#     new_rows = []
#     success_count = 0
#     error_count = 0

#     for idx, row in df.iterrows():
#         audio_path = row["audio"]
#         if not os.path.exists(audio_path):
#             logger.warning(f"Audio not found: {audio_path}")
#             error_count += 1
#             continue

#         try:
#             # 读取文件 bytes
#             with open(audio_path, "rb") as fb:
#                 block = MemoryBlock(fb.read())

#             # 解码音频
#             decoded_audio = audio_decoder(block)

#             # 检查采样率
#             sample_rate = decoded_audio["sample_rate"]
#             waveform = decoded_audio["waveform"]

#             # 如果采样率不是16000，进行重采样
#             if sample_rate != TARGET_SAMPLE_RATE:
#                 logger.info(f"Resampling {audio_path} from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
#                 waveform = resample_audio(waveform, sample_rate, TARGET_SAMPLE_RATE)
#                 # 更新decoded_audio
#                 decoded_audio = {
#                     "waveform": waveform,
#                     "sample_rate": TARGET_SAMPLE_RATE,
#                     "format": decoded_audio["format"]
#                 }

#             # 提取 fbank 特征
#             with torch.no_grad():
#                 fbank_out = fbank_converter(decoded_audio)
#                 fbank_tensor = fbank_out["fbank"]
#                 fbank_arr = fbank_tensor.cpu().numpy()

#             # 规范检查
#             if fbank_arr.size == 0:
#                 logger.error(f"Empty fbank array for {audio_path}")
#                 error_count += 1
#                 continue

#             # 检查形状是否为(time, 80)
#             if fbank_arr.shape[-1] != 80:
#                 logger.error(f"Invalid fbank shape {fbank_arr.shape} for {audio_path}, expected (time, 80)")
#                 error_count += 1
#                 continue

#             # 保存为 npy
#             audio_stem = Path(audio_path).stem
#             out_npy = output_subdir / f"{audio_stem}.npy"
#             np.save(out_npy, fbank_arr.astype(np.float32))

#             # 记录结果
#             new_rows.append({
#                 "id": row.get("id"),
#                 "audio": str(out_npy),
#                 "n_frames": fbank_arr.shape[0],
#                 "tgt_text": row.get("tgt_text"),
#                 "speaker": row.get("speaker"),
#                 "src_lang": row.get("src_lang")
#             })

#             success_count += 1

#             if (idx + 1) % 100 == 0:
#                 logger.info(f"Processed {idx+1}/{len(df)} rows (success: {success_count}, error: {error_count})")

#         except Exception as e:
#             logger.error(f"Failed feature extract for {audio_path}: {e}")
#             error_count += 1
#             continue

#     # 写出新的 npy TSV
#     if new_rows:
#         out_df = pd.DataFrame(new_rows)
#         out_name = f"{split_type}_{lang_pair}_npy.tsv"
#         out_path = TSV_OUT_DIR / out_name
#         out_df.to_csv(out_path, sep="\t", index=False)
#         logger.info(f"Saved fbank TSV: {out_path}")
#         logger.info(f"Summary: success={success_count}, error={error_count}")
#     else:
#         logger.warning(f"No valid rows extracted for {tsv_path.name}")

# # 遍历所有 TSV
# for tsv_file in tsv_files:
#     extract_features_for_file(tsv_file)

# logger.info("All done!")
