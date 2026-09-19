import torch
from transformers import SeamlessM4TModel

def sizeof_fmt(num, suffix="B"):
    # 简单的人类可读格式
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f}Y{suffix}"

def main():
    # 加载模型
    model_name = "facebook/hf-seamless-m4t-medium"
    print(f"Loading model {model_name} ...")
    model = SeamlessM4TModel.from_pretrained(model_name)
    model.eval()

    total_params = 0
    total_trainable = 0

    print("Detailed per-parameter listing:\n")
    # 输出每个参数的名字、形状和数量
    for name, param in model.named_parameters():
        count = param.numel()  # 参数数量 (元素总数) :contentReference[oaicite:0]{index=0}
        total_params += count
        if param.requires_grad:
            total_trainable += count
        print(f"{name} | shape={tuple(param.shape)} | params={count}")

    print("\n======== Summary ========")
    print(f"Total parameters       : {total_params} ({sizeof_fmt(total_params)})")
    print(f"Trainable parameters   : {total_trainable} ({sizeof_fmt(total_trainable)})")
    print(f"Non-trainable params   : {total_params - total_trainable} ({sizeof_fmt(total_params - total_trainable)})")

if __name__ == "__main__":
    main()



# import torch
# import re
# from collections import defaultdict
# from transformers import SeamlessM4TModel, AutoConfig

# # 计算参数数量的工具
# def sizeof_fmt(num, suffix=""):
#     for unit in ["", "K", "M", "B"]:
#         if abs(num) < 1000:
#             return f"{num:.2f}{unit}{suffix}"
#         num /= 1000
#     return f"{num:.2f}{suffix}"

# def group_param_name(name):
#     """
#     根据参数名提取模块分组
#     """
#     # 最简单按 prefix 取第一个部分
#     # 但对 transformer layer 我们做更精细分组
#     parts = name.split(".")
#     # speech encoder 模块通常包含 "speech_encoder"
#     if "speech_encoder" in parts:
#         idx = parts.index("speech_encoder")
#         # if next is "layers", record layer index
#         if idx + 2 < len(parts) and parts[idx+1] == "layers":
#             layer_idx = parts[idx+2]
#             group = f"speech_encoder.layer.{layer_idx}"
#         else:
#             group = "speech_encoder"
#     elif "encoder" in parts and "speech_encoder" not in parts:
#         # text encoder
#         # 把 text encoder 各层单独分组
#         if "encoder" in parts and "layers" in parts:
#             try:
#                 li = parts.index("layers")
#                 layer_idx = parts[li+1]
#                 group = f"text_encoder.layer.{layer_idx}"
#             except:
#                 group = "text_encoder"
#         else:
#             group = "text_encoder"
#     elif "decoder" in parts:
#         # decoder 各层单独分组
#         if "decoder" in parts and "layers" in parts:
#             try:
#                 li = parts.index("layers")
#                 layer_idx = parts[li+1]
#                 group = f"decoder.layer.{layer_idx}"
#             except:
#                 group = "decoder"
#         else:
#             group = "decoder"
#     else:
#         # shared / embeddings / others
#         group = parts[0]
#     return group

# def main():
#     model_name = "facebook/hf-seamless-m4t-medium"
#     print(f"Loading model {model_name} ...")
#     # 加载模型 config
#     config = AutoConfig.from_pretrained(model_name)
#     model = SeamlessM4TModel.from_pretrained(model_name, config=config)
#     model.eval()

#     grouped_total = defaultdict(int)
#     grouped_trainable = defaultdict(int)

#     # 保存 speech-encoder 每层参数 separately
#     speech_layers = defaultdict(lambda: {"total":0, "trainable":0})

#     for name, param in model.named_parameters():
#         group = group_param_name(name)
#         count = param.numel()
#         grouped_total[group] += count
#         if param.requires_grad:
#             grouped_trainable[group] += count

#         # 如果是 speech_encoder.layer.X.* .记录
#         if group.startswith("speech_encoder.layer."):
#             speech_layers[group]["total"] += count
#             if param.requires_grad:
#                 speech_layers[group]["trainable"] += count

#     print("\n===== Grouped Parameter Summary =====")
#     print(f"{'Module':40} | {'Total Params':15} | {'Trainable':15}")
#     print("-"*80)
#     for group, tot in sorted(grouped_total.items(), key=lambda x: x[1], reverse=True):
#         trainable = grouped_trainable[group]
#         print(f"{group:40} | {sizeof_fmt(tot):15} | {sizeof_fmt(trainable):15}")

#     # 提取 speech_encoder 最后两层
#     # 假定层名是 speech_encoder.layer.0, speech_encoder.layer.1, ...
#     all_speech_layer_keys = sorted(speech_layers.keys(), key=lambda x: int(x.split(".")[-1]))
#     if len(all_speech_layer_keys) >= 2:
#         last_two = all_speech_layer_keys[-2:]
#         print("\n===== Speech Encoder Last Two Layers =====")
#         for key in last_two:
#             tot = speech_layers[key]["total"]
#             tr = speech_layers[key]["trainable"]
#             print(f"{key:30} | Total: {sizeof_fmt(tot):10} | Trainable: {sizeof_fmt(tr):10}")
#     else:
#         print("\nNot enough speech encoder layers found.")

# if __name__ == "__main__":
#     main()
