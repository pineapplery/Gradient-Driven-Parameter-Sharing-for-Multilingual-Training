#!/usr/bin/env python3
"""
SeamlessM4T Medium + B1 Late Partial Decoupling 架构

完整的S2T模型，创新修改仅限于speech encoder的最后2层（10和11）：
- 层10: 完全共享，由所有样本共同训练
- 层11: 使用B1DecomposedEncoderLayer
  - FFN2: 50%共享 + 50%独立（25% g1 + 25% g2）
  - 其他部分保持原样

语言分组：
- group1 (g1): ["aeb", "est", "gle"]
- group2 (g2): ["bem"]

相比于原始SeamlessM4TMediumModel，区别仅在speech encoder的最后2层，
其他所有模块（text_encoder, text_decoder, t2u_model等）完全不变。
"""

from transformers import AutoProcessor, SeamlessM4TModel
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import sys
import logging

# 导入B1编码器组件
logger = logging.getLogger(__name__)

# 设置导入路径以找到B1_encoder_adapter
script_path = Path(__file__).resolve()
src_dir = script_path.parents[3]  # 从models往上3层到src
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from seamless_communication.models.unity.B1_encoder_adapter import (
        B1SpeechEncoderWrapper,
        B1DecomposedEncoderLayer,
        GroupSpecificFFN2,
    )
except ImportError as e:
    logger.error(f"Failed to import B1 components: {e}")
    raise


class SeamlessM4TMediumB1Model(nn.Module):
    """
    SeamlessM4T Medium 模型 + B1 Late Partial Decoupling 架构
    
    完整的S2T模型，除了speech encoder的最后2层使用B1创新外，
    其他所有模块保持与原始facebook/hf-seamless-m4t-medium完全相同。
    """

    def __init__(
        self,
        model_name: str = "facebook/hf-seamless-m4t-medium",
        num_decomposed_layers: int = 1,
        num_trainable_layers: int = 2,
        r_shared: int = 2048,
        r_group: int = 1024,
    ):
        """
        初始化SeamlessM4T Medium B1模型
        
        Args:
            model_name: 基础模型名称（默认为facebook/hf-seamless-m4t-medium）
            num_decomposed_layers: 分解层数（第11层）
            num_trainable_layers: 可训练层数（第10-11层，共2层）
            r_shared: 共享FFN的秩
            r_group: 组特定FFN的秩
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_decomposed_layers = num_decomposed_layers
        self.num_trainable_layers = num_trainable_layers
        self.r_shared = r_shared
        self.r_group = r_group
        
        # 加载完整的SeamlessM4T模型
        logger.info(f"Loading base model from {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4TModel.from_pretrained(model_name)
        
        # 验证模型结构完整性（S2T任务需要：speech_encoder + decoder）
        logger.info("\n=== Model Structure Verification ===")
        has_speech_encoder = hasattr(self.model, 'speech_encoder') and self.model.speech_encoder is not None
        has_text_decoder = hasattr(self.model, 'text_decoder') and self.model.text_decoder is not None
        has_t2u_model = hasattr(self.model, 't2u_model') and self.model.t2u_model is not None
        
        logger.info(f"✓ speech_encoder: {has_speech_encoder} ({type(self.model.speech_encoder).__name__ if has_speech_encoder else 'N/A'})")
        logger.info(f"✓ text_decoder: {has_text_decoder} ({type(self.model.text_decoder).__name__ if has_text_decoder else 'N/A'})")
        logger.info(f"✓ t2u_model: {has_t2u_model} ({type(self.model.t2u_model).__name__ if has_t2u_model else 'N/A'})")
        logger.info(f"✓ Model is complete S2T: {has_speech_encoder and has_text_decoder}")
        
        # 语言标签映射
        self.lang_tokens = {
            "aeb": 256005,
            "bem": 256025,
            "est": 256049,
            "gle": 256061,
            "eng": 256047
        }
        
        # 替换speech_encoder为B1架构
        logger.info("Replacing speech_encoder with B1SpeechEncoderWrapper")
        self._replace_speech_encoder_with_b1()
    
    def _replace_speech_encoder_with_b1(self) -> None:
        """将原始speech_encoder替换为B1SpeechEncoderWrapper"""
        base_encoder = self.model.speech_encoder
        
        # 诊断基础编码器结构
        logger.info("\n=== Diagnosing base_encoder ===")
        logger.info(f"Type: {type(base_encoder).__name__}")
        
        if hasattr(base_encoder, 'encoder'):
            logger.info(f"Has encoder attribute")
            if hasattr(base_encoder.encoder, 'layers'):
                num_layers = len(base_encoder.encoder.layers)
                logger.info(f"Number of encoder layers: {num_layers}")
        
        # 使用B1SpeechEncoderWrapper替换
        b1_encoder = B1SpeechEncoderWrapper(
            base_encoder=base_encoder,
            num_decomposed_layers=self.num_decomposed_layers,
            num_trainable_layers=self.num_trainable_layers,
            lang_groups={
                "g1": ["aeb", "est", "gle"],
                "g2": ["bem"],
            },
        )
        
        self.model.speech_encoder = b1_encoder
        self.b1_encoder = b1_encoder
        logger.info("✓ speech_encoder replaced with B1SpeechEncoderWrapper")
    
    def prepare_for_finetune(self) -> None:
        """
        为微调准备模型参数冻结。
        
        冻结策略：
        - 冻结所有非speech_encoder模块
        - 冻结speech_encoder的前10层（0-9）
        - 解冻speech_encoder的最后2层（10-11）
          - 层10：完全共享，所有参数可训练
          - 层11：使用B1DecomposedEncoderLayer，含有shared + group-specific部分
        - 冻结speech_encoder的其他部分（feature_projection, embed_positions等）
        """
        logger.info("=== Preparing Model for B1 Fine-tuning ===")
        
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 获取speech_encoder（现在是B1SpeechEncoderWrapper）
        speech_encoder = self.model.speech_encoder
        
        if isinstance(speech_encoder, B1SpeechEncoderWrapper):
            logger.info("Configuring B1SpeechEncoderWrapper for fine-tuning")
            
            # 冻结前10层（frozen_layers）
            for param in speech_encoder.frozen_layers.parameters():
                param.requires_grad = False
            logger.info("✓ Frozen layers 0-9")
            
            # 解冻层10（shared_trainable_layers）
            for param in speech_encoder.shared_trainable_layers.parameters():
                param.requires_grad = True
            logger.info("✓ Unfrozen layer 10 (shared)")
            
            # 解冻层11（decomposed_layers）- 包含shared + group-specific FFN2
            for layer in speech_encoder.decomposed_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info("✓ Unfrozen layer 11 (decomposed with B1 FFN2)")
            
            # 冻结其他部分
            for param in speech_encoder.feature_projection.parameters():
                param.requires_grad = False
            
            if speech_encoder.layer_norm is not None:
                for param in speech_encoder.layer_norm.parameters():
                    param.requires_grad = False
            
            if speech_encoder.intermediate_ffn is not None:
                for param in speech_encoder.intermediate_ffn.parameters():
                    param.requires_grad = False
            
            if speech_encoder.adapter is not None:
                for param in speech_encoder.adapter.parameters():
                    param.requires_grad = False
            
            logger.info("✓ Frozen feature_projection, layer_norm, intermediate_ffn, adapter")
        else:
            logger.warning("speech_encoder is not B1SpeechEncoderWrapper!")
        
        # 验证冻结结果
        self._verify_freeze()
    
    def _verify_freeze(self) -> None:
        """验证参数冻结情况"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info("\n=== Freeze Verification ===")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logger.info(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        
        # 显示可训练参数的层
        logger.info("\n=== Trainable Parameter Summary ===")
        trainable_by_module = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                module = name.split('.')[1] if '.' in name else name
                if module not in trainable_by_module:
                    trainable_by_module[module] = 0
                trainable_by_module[module] += param.numel()
        
        for module, count in sorted(trainable_by_module.items()):
            logger.info(f"  {module}: {count:,}")
    
    def freeze_all_except_encoder(self) -> None:
        """冻结除speech_encoder外所有参数"""
        for name, param in self.model.named_parameters():
            if not name.startswith("speech_encoder"):
                param.requires_grad = False
    
    def forward(self, input_features: torch.Tensor, tgt_lang: str = "eng", **kwargs) -> Dict:
        """
        前向传播（生成模式）
        
        Args:
            input_features: 音频特征 (batch_size, seq_len, feature_dim)
            tgt_lang: 目标语言代码
            **kwargs: 其他参数传给model.generate()
        
        Returns:
            生成的输出
        """
        # 获取目标语言 token
        lang_token_id = self.lang_tokens.get(tgt_lang, self.lang_tokens["eng"])
        
        # 准备输入
        inputs = {
            "input_features": input_features,
            "tgt_lang": tgt_lang,
            "forced_bos_token_id": lang_token_id,
        }
        inputs.update(kwargs)
        
        # 前向传播（生成模式）
        outputs = self.model.generate(**inputs)
        
        return outputs
    
    def compute_loss(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor,
        tgt_lang: str = "eng",
        source_lang: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        计算损失（训练模式）
        
        Args:
            input_features: 音频特征 (batch_size, seq_len, feature_dim)
            labels: 目标文本标签 (batch_size, seq_len)
            tgt_lang: 目标语言代码
            source_lang: 源语言代码（用于路由到对应的group FFN）
            **kwargs: 其他参数
        
        Returns:
            损失值
        """
        # 获取目标语言 token
        lang_token_id = self.lang_tokens.get(tgt_lang, self.lang_tokens["eng"])
        
        # 对于B1模型，需要将source_lang传给speech_encoder
        # 通过修改forward()方法来支持lang参数
        
        # 准备输入
        inputs = {
            "input_features": input_features,
            "labels": labels,
            "tgt_lang": tgt_lang,
            "forced_bos_token_id": lang_token_id,
        }
        inputs.update(kwargs)
        
        # 前向传播（训练模式）
        # 注：这里需要确保model能够正确处理source_lang
        outputs = self.model(**inputs)
        
        return outputs.loss
    
    def forward_with_lang(
        self,
        input_features: torch.Tensor,
        source_lang: str,
        tgt_lang: str = "eng",
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        带语言信息的前向传播（用于speech encoder编码）
        
        Args:
            input_features: 音频特征
            source_lang: 源语言代码（用于B1路由）
            tgt_lang: 目标语言
            **kwargs: 其他参数
        
        Returns:
            编码后的特征和padding mask
        """
        # 直接调用B1编码器
        if hasattr(self.model.speech_encoder, 'encode_speech'):
            return self.model.speech_encoder.encode_speech(
                seqs=input_features,
                lang=source_lang
            )
        else:
            return self.model.speech_encoder(input_features, **kwargs)
    
    @property
    def speech_encoder(self):
        """暴露speech_encoder属性（用于外部访问）"""
        return self.model.speech_encoder
    
    @speech_encoder.setter
    def speech_encoder(self, value):
        """设置speech_encoder属性"""
        self.model.speech_encoder = value
    
    @property
    def t2u_model(self):
        """暴露t2u_model属性"""
        return self.model.t2u_model if hasattr(self.model, 't2u_model') else None
    
    @t2u_model.setter
    def t2u_model(self, value):
        """设置t2u_model属性"""
        if hasattr(self.model, 't2u_model'):
            self.model.t2u_model = value
    
    @property
    def text_encoder(self):
        """暴露text_encoder属性"""
        return self.model.text_encoder if hasattr(self.model, 'text_encoder') else None
    
    @text_encoder.setter
    def text_encoder(self, value):
        """设置text_encoder属性"""
        if hasattr(self.model, 'text_encoder'):
            self.model.text_encoder = value
    
    @property
    def device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.model.parameters()).device
    
    def to(self, device) -> "SeamlessM4TMediumB1Model":
        """移动模型到指定设备"""
        self.model = self.model.to(device)
        return super().to(device)
    
    def state_dict(self):
        """返回模型状态字典"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """加载模型状态字典"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """设置评估模式"""
        self.model.eval()
        return super().eval()
    
    def parameters(self):
        """返回模型参数"""
        return self.model.parameters()
    
    def named_parameters(self, *args, **kwargs):
        """返回模型命名参数"""
        return self.model.named_parameters(*args, **kwargs)
