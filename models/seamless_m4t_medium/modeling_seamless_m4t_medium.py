
from transformers import AutoProcessor, SeamlessM4TModel
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class SeamlessM4TMediumModel(nn.Module):
    """
    SeamlessM4T Medium 模型封装类，用于 S2T 任务微调。
    支持冻结特定层，只训练 encoder 的最后 n 层。
    """

    def __init__(self, model_name="facebook/hf-seamless-m4t-medium"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4TModel.from_pretrained(model_name)

        # 语言标签映射
        self.lang_tokens = {
            "aeb": 256005,
            "bem": 256025,
            "est": 256049,
            "gle": 256061,
            "eng": 256047
        }

    def freeze_all_except_encoder(self):
        """
        冻结除 speech_encoder 以外所有参数。
        """
        for name, param in self.model.named_parameters():
            # 只保留 speech_encoder 可训练
            if not name.startswith("speech_encoder"):
                param.requires_grad = False

    def freeze_encoder_except_last_n(self, n=2):
        """
        冻结 speech_encoder 除最后 n 层外的所有参数。
        同时也冻结 encoder 的 embedding 和 projection 层。
        """
        encoder = self.model.speech_encoder
        if encoder is None:
            raise ValueError("speech_encoder not found in model.")

        # 先冻结所有 encoder 参数
        for param in encoder.parameters():
            param.requires_grad = False

        # 解冻最后 n 层
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
            layers = encoder.encoder.layers
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            raise ValueError("speech_encoder does not have attribute 'encoder.layers'.")

    def prepare_for_finetune(self, n=2):
        """
        一键冻结除 speech_encoder 外所有参数，并只解冻 speech_encoder 最后 n 层。
        确保只训练 encoder 的最后 n 层，其他所有参数（包括 embedding 和 projection）都被冻结。

        根据实际模型结构，需要冻结的模块包括：
        - shared: 共享的embedding
        - text_encoder: 文本编码器
        - text_decoder: 文本解码器
        - t2u_model: 文本到单元模型（包括vocoder）
        - lm_head: 语言模型头
        - speech_encoder.feature_projection: 特征投影层
        - speech_encoder.encoder.embed_positions: 位置编码
        - speech_encoder.encoder.dropout: dropout层
        - speech_encoder.encoder.layer_norm: 层归一化
        - speech_encoder.intermediate_ffn: 中间前馈网络
        - speech_encoder.adapter: 适配器层
        """
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 获取 speech_encoder
        speech_encoder = self.model.speech_encoder
        if speech_encoder is None:
            raise ValueError("speech_encoder not found in model.")

        # 只解冻 speech_encoder.encoder 的最后 n 层
        if hasattr(speech_encoder, "encoder") and hasattr(speech_encoder.encoder, "layers"):
            layers = speech_encoder.encoder.layers
            total_layers = len(layers)
            print(f"Total encoder layers: {total_layers}")
            print(f"Unfreezing last {n} layers: {total_layers-n} to {total_layers-1}")

            for idx, layer in enumerate(layers[-n:]):
                print(f"  Unfreezing layer {total_layers-n+idx}")
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            raise ValueError("speech_encoder does not have attribute 'encoder.layers'.")

        # 验证冻结结果
        self._verify_freeze()

    def _verify_freeze(self):
        """
        验证冻结结果，打印可训练参数信息。
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"=== Freeze Verification ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage: {100*trainable_params/total_params:.2f}%")

        print(f"=== Trainable Parameters ===")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,}")

        print(f"=== Frozen Modules ===")
        frozen_modules = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                module_name = name.split('.')[0]
                frozen_modules.add(module_name)

        for module in sorted(frozen_modules):
            print(f"  {module}")

    def forward(self, input_features, tgt_lang="eng"):
        """
        前向传播，用于 S2T 任务。

        Args:
            input_features: 音频特征 (batch_size, seq_len, feature_dim)
            tgt_lang: 目标语言代码

        Returns:
            model outputs
        """
        # 获取目标语言 token
        lang_token_id = self.lang_tokens.get(tgt_lang, self.lang_tokens["eng"])

        # 准备输入
        inputs = {
            "input_features": input_features,
            "tgt_lang": tgt_lang,
            "forced_bos_token_id": lang_token_id
        }

        # 前向传播
        outputs = self.model.generate(**inputs)

        return outputs

    def compute_loss(self, input_features, labels, tgt_lang="eng"):
        """
        计算损失，用于训练。

        Args:
            input_features: 音频特征 (batch_size, seq_len, feature_dim)
            labels: 目标文本标签 (batch_size, seq_len)
            tgt_lang: 目标语言代码

        Returns:
            loss
        """
        # 获取目标语言 token
        lang_token_id = self.lang_tokens.get(tgt_lang, self.lang_tokens["eng"])

        # 准备输入
        inputs = {
            "input_features": input_features,
            "labels": labels,
            "tgt_lang": tgt_lang,
            "forced_bos_token_id": lang_token_id
        }

        # 前向传播
        outputs = self.model(**inputs)

        return outputs.loss
