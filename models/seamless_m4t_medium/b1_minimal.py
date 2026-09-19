#!/usr/bin/env python3
"""
B1 Minimal Implementation - 最小改动实现B1架构

核心思路：
1. 只替换 encoder.layers[11].ffn2
2. 使用加性分解：W ≈ W_shared + W_group_g
3. 不修改 encoder.forward，不重写整个架构
4. 通过 thread-local context 传递语言组信息

语言分组：
- group1 (g1): ["aeb", "est", "gle"]
- group2 (g2): ["bem"]
"""

import torch
import torch.nn as nn
import threading
import traceback
from copy import deepcopy
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# 全局变量：存储当前batch的语言组（使用thread-local确保线程安全）
_thread_local = threading.local()


def set_current_group(group: str):
    """设置当前batch的语言组（在训练循环中调用）"""
    _thread_local.current_group = group


def get_current_group() -> str:
    """获取当前batch的语言组"""
    return getattr(_thread_local, 'current_group', 'g1')


# ===== 多Group方案配置 =====

# 2-Group方案（默认）：bem vs others
LANG_TO_GROUP_2 = {
    "aeb": "g1",  # group2
    "est": "g1",  # group2
    "gle": "g1",  # group2
    "bem": "g2",  # group1
}

def update_language_grouping(group_assignments_dict: Dict[str, str], num_groups: int = 2):
    """
    动态更新语言分组配置
    
    Args:
        group_assignments_dict: 语言到组的映射，如 {"aeb": "g1", "est": "g1", "gle": "g1", "bem": "g2"}
        num_groups: 分组数量（2或4）
    """
    global LANG_TO_GROUP_2, LANG_TO_GROUP_4
    
    if num_groups == 2:
        LANG_TO_GROUP_2.update(group_assignments_dict)
        logger.info(f"✓ Updated LANG_TO_GROUP_2: {LANG_TO_GROUP_2}")
    elif num_groups == 4:
        LANG_TO_GROUP_4.update(group_assignments_dict)
        logger.info(f"✓ Updated LANG_TO_GROUP_4: {LANG_TO_GROUP_4}")
    else:
        logger.warning(f"Unknown num_groups: {num_groups}")

# 4-Group方案（新增）：每个语言单独为一组
LANG_TO_GROUP_4 = {
    "aeb": "g0",  # group 0 - aeb 24.45%
    "bem": "g1",  # group 1 - bem 27.68%
    "est": "g2",  # group 2 - est 26.26%
    "gle": "g3",  # group 3 - gle 21.61%
}

# 4-Group方案的能量占比（残差初始化用）
ENERGY_RATIOS_4 = {
    "g0": 0.2445,  # aeb
    "g1": 0.2768,  # bem
    "g2": 0.2626,  # est
    "g3": 0.2161,  # gle
}


def get_group_for_lang(lang: str, num_groups: int = 2) -> str:
    """根据语言代码和group数量获取对应的组"""
    if num_groups == 2:
        return LANG_TO_GROUP_2.get(lang, "g1")
    elif num_groups == 4:
        return LANG_TO_GROUP_4.get(lang, "g0")
    else:
        logger.warning(f"Unknown num_groups {num_groups}, using 2-group scheme")
        return LANG_TO_GROUP_2.get(lang, "g1")


def _get_encoder_ffn2_module(speech_encoder):
    """获取encoder.layers[11].ffn2模块（原始逻辑）"""
    # 获取 encoder.layers（支持多种模型结构）
    layers = None
    
    if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
        layers = speech_encoder.encoder.layers
        layers_path = "speech_encoder.encoder.layers"
    elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
        layers = speech_encoder.inner.layers
        layers_path = "speech_encoder.inner.layers"
    elif hasattr(speech_encoder, 'layers'):
        layers = speech_encoder.layers
        layers_path = "speech_encoder.layers"
    else:
        logger.error("Cannot find encoder layers. Available attributes:")
        for attr in dir(speech_encoder):
            if not attr.startswith('_'):
                logger.error(f"  - {attr}")
        raise ValueError("Cannot find encoder.layers in speech_encoder")
    
    total_layers = len(layers)
    if total_layers < 12:
        raise ValueError(f"Expected at least 12 layers, got {total_layers}")
    
    # 深拷贝第11层
    layer11 = deepcopy(layers[11])
    
    # 查找原始ffn2
    original_ffn2 = None
    ffn2_attr_name = None
    
    for attr_name in ['ffn2', 'ffn', 'feed_forward', 'mlp']:
        if hasattr(layer11, attr_name):
            original_ffn2 = getattr(layer11, attr_name)
            ffn2_attr_name = attr_name
            break
    
    if original_ffn2 is None:
        raise ValueError("Cannot find ffn2 in layer 11. Available attributes: " + 
                        ", ".join([attr for attr in dir(layer11) if not attr.startswith('_')]))
    
    return f"{layers_path}[11].{ffn2_attr_name}", original_ffn2, ffn2_attr_name


def _get_encoder_ffn2_layer10_module(speech_encoder):
    """获取encoder.layers[10].ffn2模块（第10层，新增功能）"""
    # 获取 encoder.layers（支持多种模型结构）
    layers = None
    
    if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
        layers = speech_encoder.encoder.layers
        layers_path = "speech_encoder.encoder.layers"
    elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
        layers = speech_encoder.inner.layers
        layers_path = "speech_encoder.inner.layers"
    elif hasattr(speech_encoder, 'layers'):
        layers = speech_encoder.layers
        layers_path = "speech_encoder.layers"
    else:
        logger.error("Cannot find encoder layers. Available attributes:")
        for attr in dir(speech_encoder):
            if not attr.startswith('_'):
                logger.error(f"  - {attr}")
        raise ValueError("Cannot find encoder.layers in speech_encoder")
    
    total_layers = len(layers)
    if total_layers < 11:
        raise ValueError(f"Expected at least 11 layers, got {total_layers}")
    
    # 深拷贝第10层
    layer10 = deepcopy(layers[10])
    
    # 查找原始ffn2
    original_ffn2 = None
    ffn2_attr_name = None
    
    for attr_name in ['ffn2', 'ffn', 'feed_forward', 'mlp']:
        if hasattr(layer10, attr_name):
            original_ffn2 = getattr(layer10, attr_name)
            ffn2_attr_name = attr_name
            break
    
    if original_ffn2 is None:
        raise ValueError("Cannot find ffn2 in layer 10. Available attributes: " + 
                        ", ".join([attr for attr in dir(layer10) if not attr.startswith('_')]))
    
    return f"{layers_path}[10].{ffn2_attr_name}", original_ffn2, ffn2_attr_name


def _get_adapter_ffn_module(speech_encoder):
    """获取adapter.layers[0].ffn模块"""
    logger.info("Searching for adapter module...")
    
    # 检查多种可能的adapter位置
    adapter_paths = [
        ('adaptor_layers', 'speech_encoder.adaptor_layers'),  # SeamlessM4T uses adaptor_layers
        ('adapter', 'speech_encoder.adapter'),
        ('adaptor', 'speech_encoder.adaptor'),  # 可能的拼写变体
        ('inner.adapter', 'speech_encoder.inner.adapter'),
        ('inner.adaptor', 'speech_encoder.inner.adaptor'),
        ('inner.adaptor_layers', 'speech_encoder.inner.adaptor_layers'),
    ]
    
    adapter = None
    adapter_path = None
    
    for attr_path, description in adapter_paths:
        try:
            # 支持嵌套属性访问
            obj = speech_encoder
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr)
            adapter = obj
            adapter_path = description
            logger.info(f"✓ Found adapter at: {description}")
            break
        except AttributeError:
            logger.debug(f"No adapter found at: {description}")
            continue
    
    if adapter is None:
        # 提供详细的调试信息
        logger.error("Failed to find adapter module. Available attributes:")
        for attr in dir(speech_encoder):
            if not attr.startswith('_'):
                logger.error(f"  - speech_encoder.{attr}")
        if hasattr(speech_encoder, 'inner'):
            logger.error("speech_encoder.inner attributes:")
            for attr in dir(speech_encoder.inner):
                if not attr.startswith('_'):
                    logger.error(f"  - speech_encoder.inner.{attr}")
        raise ValueError("Cannot find adapter module in speech_encoder. "
                        "Check the model structure. Available target modules: encoder_ffn1, encoder_ffn2, intermediate_ffn")
    
    # 检查adapter是否是列表（adaptor_layers直接是层列表）或有layers属性
    if hasattr(adapter, '__getitem__') and hasattr(adapter, '__len__') and not hasattr(adapter, 'layers'):
        # adaptor_layers 是直接的层列表
        if len(adapter) == 0:
            raise ValueError(f"Adaptor layers at {adapter_path} is empty")
        adapter_layer0 = adapter[0]
        layer_path = f"{adapter_path}[0]"
        logger.info(f"✓ Using direct adaptor_layers structure: {layer_path}")
    elif hasattr(adapter, 'layers'):
        # 传统的adapter.layers结构
        if len(adapter.layers) == 0:
            raise ValueError(f"Adapter at {adapter_path} does not have layers or layers is empty")
        adapter_layer0 = adapter.layers[0]
        layer_path = f"{adapter_path}.layers[0]"
        logger.info(f"✓ Using adapter.layers structure: {layer_path}")
    else:
        raise ValueError(f"Adapter at {adapter_path} does not have expected structure (layers list or direct list)")
    
    # 查找FFN模块
    ffn = None
    ffn_attr_name = None
    
    for attr_name in ['ffn', 'feed_forward', 'mlp']:
        if hasattr(adapter_layer0, attr_name):
            ffn = getattr(adapter_layer0, attr_name)
            ffn_attr_name = attr_name
            logger.info(f"✓ Found FFN at: {layer_path}.{ffn_attr_name}")
            break
    
    if ffn is None:
        available_attrs = [attr for attr in dir(adapter_layer0) if not attr.startswith('_')]
        raise ValueError(f"Adapter layer 0 at {layer_path} does not have FFN attribute. "
                        f"Available attributes: {available_attrs}")
    
    return f"{layer_path}.{ffn_attr_name}", ffn, ffn_attr_name


def _get_encoder_ffn1_module(speech_encoder):
    """获取encoder.layers[11].ffn1模块"""
    # 获取 encoder.layers（支持多种模型结构）
    layers = None
    
    if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
        layers = speech_encoder.encoder.layers
        layers_path = "speech_encoder.encoder.layers"
    elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
        layers = speech_encoder.inner.layers
        layers_path = "speech_encoder.inner.layers"
    elif hasattr(speech_encoder, 'layers'):
        layers = speech_encoder.layers
        layers_path = "speech_encoder.layers"
    else:
        logger.error("Cannot find encoder layers. Available attributes:")
        for attr in dir(speech_encoder):
            if not attr.startswith('_'):
                logger.error(f"  - {attr}")
        raise ValueError("Cannot find encoder.layers in speech_encoder")
    
    total_layers = len(layers)
    if total_layers < 12:
        raise ValueError(f"Expected at least 12 layers, got {total_layers}")
    
    # 深拷贝第11层
    layer11 = deepcopy(layers[11])
    
    # 查找原始ffn1
    original_ffn1 = None
    ffn1_attr_name = None
    
    for attr_name in ['ffn1', 'ffn', 'feed_forward_1', 'mlp1']:
        if hasattr(layer11, attr_name):
            original_ffn1 = getattr(layer11, attr_name)
            ffn1_attr_name = attr_name
            break
    
    if original_ffn1 is None:
        raise ValueError("Cannot find ffn1 in layer 11. Available attributes: " + 
                        ", ".join([attr for attr in dir(layer11) if not attr.startswith('_')]))
    
    return f"{layers_path}[11].{ffn1_attr_name}", original_ffn1, ffn1_attr_name


def _get_encoder_layer10_ffn2_module(speech_encoder):
    """获取encoder.layers[10].ffn2模块（新增：第十层）"""
    # 获取 encoder.layers（支持多种模型结构）
    layers = None
    
    if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
        layers = speech_encoder.encoder.layers
        layers_path = "speech_encoder.encoder.layers"
    elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
        layers = speech_encoder.inner.layers
        layers_path = "speech_encoder.inner.layers"
    elif hasattr(speech_encoder, 'layers'):
        layers = speech_encoder.layers
        layers_path = "speech_encoder.layers"
    else:
        logger.error("Cannot find encoder layers. Available attributes:")
        for attr in dir(speech_encoder):
            if not attr.startswith('_'):
                logger.error(f"  - speech_encoder.{attr}")
        raise ValueError("Cannot find encoder.layers in speech_encoder")
    
    total_layers = len(layers)
    if total_layers < 11:
        raise ValueError(f"Expected at least 11 layers, got {total_layers}")
    
    # 深拷贝第10层
    layer10 = deepcopy(layers[10])
    
    # 查找原始ffn2
    original_ffn2 = None
    ffn2_attr_name = None
    
    for attr_name in ['ffn2', 'ffn', 'feed_forward', 'mlp']:
        if hasattr(layer10, attr_name):
            original_ffn2 = getattr(layer10, attr_name)
            ffn2_attr_name = attr_name
            logger.info(f"✓ Found ffn2 at: {layers_path}[10].{attr_name}")
            break
    
    if original_ffn2 is None:
        raise ValueError("Cannot find ffn2 in layer 10. Available attributes: " + 
                        ", ".join([attr for attr in dir(layer10) if not attr.startswith('_')]))
    
    # ===== [DIAG] 打印 layer10.ffn2 权重数值，确认 deepcopy 取到了预训练权重而非随机初始化 =====
    logger.info("[DIAG] layer10.ffn2 weight statistics (should NOT be near Kaiming random):")
    if hasattr(original_ffn2, 'inner_proj'):
        w1 = original_ffn2.inner_proj.weight.data.float()
        w2 = original_ffn2.output_proj.weight.data.float()
        logger.info(f"  [DIAG] inner_proj.weight: shape={w1.shape} mean={w1.mean():.4e} std={w1.std():.4e} abs_max={w1.abs().max():.4e}")
        logger.info(f"  [DIAG] output_proj.weight: shape={w2.shape} mean={w2.mean():.4e} std={w2.std():.4e} abs_max={w2.abs().max():.4e}")
        kaiming_std = (2.0 / 1024) ** 0.5  # ~0.044
        if abs(w1.std().item() - kaiming_std) < 0.01:
            logger.warning(f"  [DIAG] ⚠️  inner_proj std≈{w1.std():.4f} ≈ Kaiming({kaiming_std:.4f}) → weights may NOT be from checkpoint!")
        else:
            logger.info(f"  [DIAG] ✓ inner_proj std differs from Kaiming ({kaiming_std:.4f}) → pretrained weights confirmed")
    elif hasattr(original_ffn2, 'intermediate_dense'):
        w1 = original_ffn2.intermediate_dense.weight.data.float()
        logger.info(f"  [DIAG] intermediate_dense.weight: shape={w1.shape} mean={w1.mean():.4e} std={w1.std():.4e}")
    else:
        params = list(original_ffn2.parameters())
        if params:
            p = params[0].data.float()
            logger.info(f"  [DIAG] first param: shape={p.shape} mean={p.mean():.4e} std={p.std():.4e}")
        else:
            logger.warning("  [DIAG] ⚠️  original_ffn2 has NO parameters!")
    # ===== [DIAG END] =====
    
    return f"{layers_path}[10].{ffn2_attr_name}", original_ffn2, ffn2_attr_name


def _get_intermediate_ffn_module(speech_encoder):
    """获取intermediate_ffn模块 - SeamlessM4T模型中确实存在此模块"""
    logger.info("Searching for intermediate_ffn module...")
    
    # 检查 speech_encoder.intermediate_ffn（根据用户的模型架构输出）
    if hasattr(speech_encoder, 'intermediate_ffn'):
        original_ffn = speech_encoder.intermediate_ffn
        logger.info("✓ Found intermediate_ffn at: speech_encoder.intermediate_ffn")
        return [], original_ffn, 'intermediate_ffn'
    
    # 如果找不到，提供详细的错误信息
    logger.error("Failed to find intermediate_ffn module. Available attributes:")
    for attr in sorted(dir(speech_encoder)):
        if not attr.startswith('_'):
            logger.error(f"  - speech_encoder.{attr}")
    
    if hasattr(speech_encoder, 'inner'):
        logger.error("speech_encoder.inner attributes:")
        for attr in sorted(dir(speech_encoder.inner)):
            if not attr.startswith('_'):
                logger.error(f"  - speech_encoder.inner.{attr}")
    
    raise ValueError("Cannot find intermediate_ffn module in speech_encoder. "
                    "Available target modules: encoder_ffn1, encoder_ffn2, adapter_ffn")


class GroupSpecificFFN2(nn.Module):
    """
    加性分解的FFN2（Additive Decomposition）- 支持可配置的group数量
    
    核心思想：W ≈ W_shared + sum(W_group_i)
    
    2-Group方案（默认）:
    - shared_ffn2: 1024 → r_shared → 1024  (2048, 50%)
    - group1_ffn2: 1024 → r_group1 → 1024  (1024, bem 25%)
    - group2_ffn2: 1024 → r_group2 → 1024  (1024, others 25%)
    
    4-Group方案:
    - shared_ffn2: 1024 → r_shared → 1024  (2048, 50%)
    - group0_ffn2: 1024 → r_group_4 → 1024 (512, aeb 24.45%)
    - group1_ffn2: 1024 → r_group_4 → 1024 (512, bem 27.68%)
    - group2_ffn2: 1024 → r_group_4 → 1024 (512, est 26.26%)
    - group3_ffn2: 1024 → r_group_4 → 1024 (512, gle 21.61%)
    
    前向传播：
    y = shared_ffn2(x) + group_ffn2(x)
    
    初始化策略：
    - shared_ffn2：从原始FFN的SVD主成分初始化
    - group_ffn2：初始化为小随机噪声（std=1e-4）或Kaiming
    """
    
    def __init__(
        self,
        model_dim: int = 1024,
        hidden_dim_original: int = 4096,
        share_ratio: float = 0.5,  # 新增：共享比例，控制 r_shared 和 k_shared
        pretrained_ffn2: Optional[nn.Module] = None,
        noise_std: float = 1e-4,
        dropout_rate: float = 0.0,
        g2_energy_ratio: float = 0.2768,  # 2-group方案：bem能量占比(top10:0.2768, top20:0.2729, top5:0.2785)
        group_energy_ratios: Optional[Dict[str, float]] = None,  # 新增：动态能量比例配置，如 {"g1": 0.723, "g2": 0.277}
        init_strategy: str = 'residual',
        num_groups: int = 2,  # 新增：group数量 (2 或 4)
        use_noise_init: bool = False,  # 新增：扩展部分是否使用小噪声初始化（默认补0）
    ):
        super().__init__()
        
        self.num_groups = num_groups
        self.model_dim = model_dim
        self.hidden_dim_original = hidden_dim_original
        self.share_ratio = share_ratio
        self.use_noise_init = use_noise_init
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.init_strategy = init_strategy
        self.pretrained_w1 = None
        self.pretrained_w2 = None
        self.W_equiv_pretrained = None  # 修改：用于残差初始化的等效权重矩阵
        self._route_stats = {}
        
        # 基于 share_ratio 计算秩参数
        self.k_shared = int(model_dim * share_ratio)  # 奇异值数量
        self.r_shared = int(hidden_dim_original * share_ratio)  # 架构中间维度
        self.k_remaining = model_dim - self.k_shared
        self.r_remaining = hidden_dim_original - self.r_shared
        
        logger.info(f"Share ratio: {share_ratio}")
        logger.info(f"  k_shared (SVD rank): {self.k_shared}")
        logger.info(f"  r_shared (architecture): {self.r_shared}")
        logger.info(f"  k_remaining: {self.k_remaining}")
        logger.info(f"  r_remaining: {self.r_remaining}")
        
        # 处理动态能量比例配置
        if group_energy_ratios is not None:
            # 使用传入的能量比例覆盖默认值
            if num_groups == 2 and "g2" in group_energy_ratios:
                g2_energy_ratio = group_energy_ratios["g2"]
                logger.info(f"Using custom g2_energy_ratio: {g2_energy_ratio}")
            elif num_groups == 4:
                # 更新4-group的能量比例
                global ENERGY_RATIOS_4
                ENERGY_RATIOS_4.update(group_energy_ratios)
                logger.info(f"Using custom 4-group energy ratios: {group_energy_ratios}")
        
        # ===== 根据num_groups配置参数 =====
        if num_groups == 2:
            # 2-Group方案
            self.g2_energy_ratio = g2_energy_ratio
            
            # 归一化检查
            g1_ratio = 1 - g2_energy_ratio
            total_ratio = g1_ratio + g2_energy_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                logger.warning(f"Energy ratios sum to {total_ratio:.6f}, not 1.0. Normalizing...")
                self.g2_energy_ratio = g2_energy_ratio / total_ratio
            
            # 计算每个 group 的秩
            self.k_group1 = int(self.k_remaining * self.g2_energy_ratio)  # bem
            self.k_group2 = self.k_remaining - self.k_group1  # others
            self.r_group1 = self.r_remaining // 2
            self.r_group2 = self.r_remaining - self.r_group1
            
            logger.info(f"2-Group k allocation:")
            logger.info(f"  k_group1 (bem): {self.k_group1} ({100*self.g2_energy_ratio:.2f}%)")
            logger.info(f"  k_group2 (others): {self.k_group2} ({100*(1-self.g2_energy_ratio):.2f}%)")
            logger.info(f"  r_group1: {self.r_group1}, r_group2: {self.r_group2}")
            
            self._route_stats = {"bem": 0, "other": 0}
            
        elif num_groups == 4:
            # 4-Group方案 - 归一化检查
            total_ratio = sum(ENERGY_RATIOS_4.values())
            if abs(total_ratio - 1.0) > 1e-6:
                logger.warning(f"4-Group energy ratios sum to {total_ratio:.6f}, not 1.0. Normalizing...")
                for key in ENERGY_RATIOS_4:
                    ENERGY_RATIOS_4[key] /= total_ratio
            
            # 计算每个 group 的 k 值（按能量比例）
            self.k_group0 = int(self.k_remaining * ENERGY_RATIOS_4['g0'])  # aeb
            self.k_group1 = int(self.k_remaining * ENERGY_RATIOS_4['g1'])  # bem
            self.k_group2 = int(self.k_remaining * ENERGY_RATIOS_4['g2'])  # est
            self.k_group3 = self.k_remaining - (self.k_group0 + self.k_group1 + self.k_group2)  # gle (补齐)
            
            # 计算每个 group 的 r 值（平均分配）
            self.r_group_4 = self.r_remaining // 4
            self.r_group1 = self.r_group_4
            self.r_group2 = self.r_group_4
            
            logger.info(f"4-Group k allocation:")
            logger.info(f"  k_group0 (aeb): {self.k_group0} ({100*ENERGY_RATIOS_4['g0']:.2f}%)")
            logger.info(f"  k_group1 (bem): {self.k_group1} ({100*ENERGY_RATIOS_4['g1']:.2f}%)")
            logger.info(f"  k_group2 (est): {self.k_group2} ({100*ENERGY_RATIOS_4['g2']:.2f}%)")
            logger.info(f"  k_group3 (gle): {self.k_group3} ({100*ENERGY_RATIOS_4['g3']:.2f}%)")
            logger.info(f"  r_group_4: {self.r_group_4}")
            
            self._route_stats = {"aeb": 0, "bem": 0, "est": 0, "gle": 0}
            
        else:
            raise ValueError(f"num_groups must be 2 or 4, got {num_groups}")
        
        # Shared FFN2
        self.shared_ffn2 = nn.Sequential(
            nn.Linear(model_dim, self.r_shared, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.r_shared, model_dim, bias=True),
        )
        
        # Group-specific FFN2s
        if num_groups == 2:
            # 2-Group: bem vs others
            self.group1_ffn2 = nn.Sequential(
                nn.Linear(model_dim, self.r_group1, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.r_group1, model_dim, bias=True),
            )
            
            self.group2_ffn2 = nn.Sequential(
                nn.Linear(model_dim, self.r_group2, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.r_group2, model_dim, bias=True),
            )
            
            self.group_ffn2_list = [self.group1_ffn2, self.group2_ffn2]
            
        elif num_groups == 4:
            # 4-Group: aeb, bem, est, gle各自独立
            self.group0_ffn2 = nn.Sequential(  # aeb
                nn.Linear(model_dim, self.r_group_4, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.r_group_4, model_dim, bias=True),
            )
            
            self.group1_ffn2 = nn.Sequential(  # bem
                nn.Linear(model_dim, self.r_group_4, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.r_group_4, model_dim, bias=True),
            )
            
            self.group2_ffn2 = nn.Sequential(  # est
                nn.Linear(model_dim, self.r_group_4, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.r_group_4, model_dim, bias=True),
            )
            
            self.group3_ffn2 = nn.Sequential(  # gle
                nn.Linear(model_dim, self.r_group_4, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.r_group_4, model_dim, bias=True),
            )
            
            self.group_ffn2_list = [self.group0_ffn2, self.group1_ffn2, self.group2_ffn2, self.group3_ffn2]
        
        # 初始化
        if pretrained_ffn2 is not None:
            self._init_shared_from_pretrained(pretrained_ffn2)
            
            if self.init_strategy == 'residual':
                self._init_group_from_residual()
            elif self.init_strategy == 'random':
                self._init_group_with_kaiming()
            else:
                logger.warning(f"Unknown init_strategy '{self.init_strategy}', using 'residual'")
                self._init_group_from_residual()
        else:
            self._init_group_with_noise()
        
        # 标记参数属性shared或group（用于分组学习率）
        for param in self.shared_ffn2.parameters():
            param.param_group = 'shared'
        for group_ffn2 in self.group_ffn2_list:
            for param in group_ffn2.parameters():
                param.param_group = 'group'
        
        # 日志
        if num_groups == 2:
            logger.info(f"✓ GroupSpecificFFN2 initialized (2-Group mode):")
            logger.info(f"  - Shared: {model_dim} → {self.r_shared} → {model_dim} ({100*share_ratio:.1f}%)") 
            logger.info(f"  - Group1 (bem):  {model_dim} → {self.r_group1} → {model_dim} (k={self.k_group1}, energy {100*g2_energy_ratio:.2f}%)")
            logger.info(f"  - Group2 (others): {model_dim} → {self.r_group2} → {model_dim} (k={self.k_group2}, energy {100*(1-g2_energy_ratio):.2f}%)")
        else:
            logger.info(f"✓ GroupSpecificFFN2 initialized (4-Group mode):")
            logger.info(f"  - Shared: {model_dim} → {self.r_shared} → {model_dim} ({100*share_ratio:.1f}%)")
            logger.info(f"  - Group0 (aeb):  {model_dim} → {self.r_group_4} → {model_dim} (k={self.k_group0}, energy {100*ENERGY_RATIOS_4['g0']:.2f}%)")
            logger.info(f"  - Group1 (bem):  {model_dim} → {self.r_group_4} → {model_dim} (k={self.k_group1}, energy {100*ENERGY_RATIOS_4['g1']:.2f}%)")
            logger.info(f"  - Group2 (est):  {model_dim} → {self.r_group_4} → {model_dim} (k={self.k_group2}, energy {100*ENERGY_RATIOS_4['g2']:.2f}%)")
            logger.info(f"  - Group3 (gle):  {model_dim} → {self.r_group_4} → {model_dim} (k={self.k_group3}, energy {100*ENERGY_RATIOS_4['g3']:.2f}%)")
        logger.info(f"  - Initialization strategy: {init_strategy}")
        logger.info(f"  - Use noise init: {use_noise_init}")
    
    def _init_shared_from_pretrained(self, pretrained_ffn2: nn.Module) -> None:
        """
        从预训练FFN初始化shared_ffn2（一次SVD + 对称因子分解）
        
        新策略：
        1. 计算等效矩阵 W_equiv = (w2 @ w1).T，维度 (1024, 1024)
        2. SVD分解 W_equiv → U, S, Vh
        3. 提取前 k_shared 个奇异值做对称因子分解
        4. 扩展到 r_shared 维度，多余部分补零或小噪声
        5. 保存 W_equiv 用于后续残差计算
        """
        try:
            # 查找原始FFN的权重
            w1, w2, b1, b2 = None, None, None, None
            
            # 尝试1: fairseq2 StandardFeedForwardNetwork (inner_proj, output_proj)
            if hasattr(pretrained_ffn2, 'inner_proj') and hasattr(pretrained_ffn2, 'output_proj'):
                w1 = pretrained_ffn2.inner_proj.weight.data  # (4096, 1024)
                b1 = pretrained_ffn2.inner_proj.bias.data if hasattr(pretrained_ffn2.inner_proj, 'bias') and pretrained_ffn2.inner_proj.bias is not None else None
                w2 = pretrained_ffn2.output_proj.weight.data  # (1024, 4096)
                b2 = pretrained_ffn2.output_proj.bias.data if hasattr(pretrained_ffn2.output_proj, 'bias') and pretrained_ffn2.output_proj.bias is not None else None
                logger.info(f"✓ Found fairseq2 StandardFeedForwardNetwork")
                logger.info(f"  w1 shape: {w1.shape}, w2 shape: {w2.shape}")
            
            # 尝试2: HuggingFace SeamlessM4TConformerFeedForward (intermediate_dense, output_dense)
            elif hasattr(pretrained_ffn2, 'intermediate_dense') and hasattr(pretrained_ffn2, 'output_dense'):
                w1 = pretrained_ffn2.intermediate_dense.weight.data  # (4096, 1024)
                b1 = pretrained_ffn2.intermediate_dense.bias.data if pretrained_ffn2.intermediate_dense.bias is not None else None
                w2 = pretrained_ffn2.output_dense.weight.data  # (1024, 4096)
                b2 = pretrained_ffn2.output_dense.bias.data if pretrained_ffn2.output_dense.bias is not None else None
                logger.info(f"✓ Found HuggingFace SeamlessM4TConformerFeedForward (intermediate_dense/output_dense)")
                logger.info(f"  w1 shape: {w1.shape}, w2 shape: {w2.shape}")
            
            # 尝试3: HuggingFace Sequential
            elif isinstance(pretrained_ffn2, nn.Sequential):
                for module in pretrained_ffn2:
                    if isinstance(module, nn.Linear):
                        if w1 is None:
                            w1 = module.weight.data  # shape: (hidden_dim, model_dim)
                            b1 = module.bias.data if module.bias is not None else None
                        else:
                            w2 = module.weight.data  # shape: (model_dim, hidden_dim)
                            b2 = module.bias.data if module.bias is not None else None
            
            # 尝试4: 单个Linear层
            elif hasattr(pretrained_ffn2, 'weight'):
                w1 = pretrained_ffn2.weight.data
                b1 = pretrained_ffn2.bias.data if pretrained_ffn2.bias is not None else None
            
            if w1 is not None and w2 is not None:
                # Step 1: 计算等效矩阵 W_equiv = (w2 @ w1).T
                # w2: (1024, 4096), w1: (4096, 1024) → w2 @ w1: (1024, 1024)
                logger.info(f"Step 1: Computing equivalent matrix W_equiv = (w2 @ w1).T")
                logger.info(f"  w1 shape: {w1.shape}, w2 shape: {w2.shape}")
                
                W_equiv = (w2 @ w1).T  # (1024, 1024)
                logger.info(f"  W_equiv shape: {W_equiv.shape}")
                
                # 保存用于残差计算
                self.W_equiv_pretrained = W_equiv.clone().detach()
                self.pretrained_w1 = w1.clone().detach()  # 保留以备兼容性
                self.pretrained_w2 = w2.clone().detach()  # 保留以备兼容性
                
                # Step 2: SVD 分解
                W_equiv_fp32 = W_equiv.to(dtype=torch.float32, device='cpu')
                logger.info(f"Step 2: Performing SVD on W_equiv ({W_equiv_fp32.shape})")
                U, S, Vh = torch.linalg.svd(W_equiv_fp32, full_matrices=False)
                # U: (1024, 1024), S: (1024,), Vh: (1024, 1024)
                logger.info(f"  SVD results: U={U.shape}, S={S.shape}, Vh={Vh.shape}")
                logger.info(f"  Top-5 singular values: {S[:5].tolist()}")
                
                # Step 3: 计算对称因子（一次SVD）
                k_shared = self.k_shared  # 已经在__init__中计算
                logger.info(f"Step 3: Extracting top k_shared={k_shared} singular values")
                
                sqrt_S = torch.sqrt(S[:k_shared])
                W1_factor = U[:, :k_shared] @ torch.diag(sqrt_S)  # (1024, k_shared)
                W2_factor = torch.diag(sqrt_S) @ Vh[:k_shared, :]  # (k_shared, 1024)
                logger.info(f"  W1_factor shape: {W1_factor.shape}, W2_factor shape: {W2_factor.shape}")
                
                # Step 4: 扩展到 r_shared 维度
                r_shared = self.r_shared
                logger.info(f"Step 4: Expanding to r_shared={r_shared}")
                
                # W1: (1024, k_shared) → (1024, r_shared)
                W1_expanded = torch.zeros(1024, r_shared, dtype=W1_factor.dtype, device=W1_factor.device)
                W1_expanded[:, :k_shared] = W1_factor
                if self.use_noise_init and k_shared < r_shared:
                    W1_expanded[:, k_shared:] += torch.randn_like(W1_expanded[:, k_shared:]) * self.noise_std
                    logger.info(f"  Added noise to W1_expanded[:, {k_shared}:]")
                
                # W2: (k_shared, 1024) → (r_shared, 1024)
                W2_expanded = torch.zeros(r_shared, 1024, dtype=W2_factor.dtype, device=W2_factor.device)
                W2_expanded[:k_shared, :] = W2_factor
                if self.use_noise_init and k_shared < r_shared:
                    W2_expanded[k_shared:, :] += torch.randn_like(W2_expanded[k_shared:, :]) * self.noise_std
                    logger.info(f"  Added noise to W2_expanded[{k_shared}:, :]")
                
                # Step 5: 转置赋值给 shared_ffn2
                # shared_ffn2[0]: Linear(1024, r_shared) → weight.shape = (r_shared, 1024)
                # shared_ffn2[3]: Linear(r_shared, 1024) → weight.shape = (1024, r_shared)
                logger.info(f"Step 5: Assigning to shared_ffn2")
                self.shared_ffn2[0].weight.data.copy_(
                    W1_expanded.T.to(dtype=w1.dtype, device=w1.device)
                )
                if b1 is not None:
                    # 取前 r_shared 个bias
                    bias_len = min(len(b1), self.r_shared)
                    self.shared_ffn2[0].bias.data[:bias_len].copy_(b1[:bias_len])
                
                self.shared_ffn2[3].weight.data.copy_(
                    W2_expanded.T.to(dtype=w2.dtype, device=w2.device)
                )
                if b2 is not None:
                    self.shared_ffn2[3].bias.data.copy_(b2)
                
                logger.info(f"✓ Shared FFN2 initialized successfully")
                logger.info(f"  k_shared (SVD rank): {k_shared}")
                logger.info(f"  r_shared (architecture): {r_shared}")
                logger.info(f"  Strategy: One-shot SVD + symmetric factorization")
                # ===== [DIAG] 验证shared_ffn2权重数值合理性 =====
                w_in  = self.shared_ffn2[0].weight.data.float()
                w_out = self.shared_ffn2[3].weight.data.float()
                logger.info(f"  [DIAG] shared_ffn2[0].weight: shape={w_in.shape} mean={w_in.mean():.4e} std={w_in.std():.4e} abs_max={w_in.abs().max():.4e}")
                logger.info(f"  [DIAG] shared_ffn2[3].weight: shape={w_out.shape} mean={w_out.mean():.4e} std={w_out.std():.4e} abs_max={w_out.abs().max():.4e}")
                # 重建误差：W_reconstructed = (shared_ffn2[0].T) @ (shared_ffn2[3].T)
                W_recon = w_in.T @ w_out.T  # (1024, 1024)
                W_orig  = (w2.float() @ w1.float()).T  # (1024, 1024)
                recon_err = (W_orig - W_recon).norm().item()
                orig_norm = W_orig.norm().item()
                logger.info(f"  [DIAG] W_equiv norm={orig_norm:.4f}, W_shared_recon norm={W_recon.norm().item():.4f}")
                logger.info(f"  [DIAG] Reconstruction error ||W_orig - W_shared||={recon_err:.4f}  ({100*recon_err/max(orig_norm,1e-8):.2f}% of ||W_orig||)")
                logger.info(f"  [DIAG] Top-3 singular values captured: {S[:3].tolist()}")
                logger.info(f"  [DIAG] Energy captured by top-{k_shared} SVs: {(S[:k_shared]**2).sum().item()/((S**2).sum().item()+1e-10)*100:.2f}%")
                # ===== [DIAG END] =====
            else:
                logger.warning("Could not find weights in pretrained_ffn2, using random init")
                logger.warning("  [DIAG] ⚠️  w1 or w2 is None - this means FFN format was not recognized")
                logger.warning(f"  [DIAG] pretrained_ffn2 type: {type(pretrained_ffn2).__name__}")
                logger.warning(f"  [DIAG] pretrained_ffn2 attrs: {[a for a in dir(pretrained_ffn2) if not a.startswith('_')][:15]}")
        
        except Exception as e:
            logger.warning(f"Failed to initialize from pretrained weights: {e}")
            logger.warning("Using random initialization instead")
            import traceback
            logger.warning(traceback.format_exc())
    
    def _init_group_from_residual(self) -> None:
        """
        从统一权重矩阵的残差初始化 group FFN 参数
        
        策略：
        1. 重建 W_shared = Ain.T @ Aout.T
        2. 计算 residual = W_unify - W_shared
        3. 按能量比例分配残差给各个 group
        4. 对每个 group 的残差做低秩分解
        
        2-Group:
           - g1 (bem): g2_energy_ratio * residual
           - g2 (others): (1 - g2_energy_ratio) * residual
        
        4-Group:
           - g0 (aeb):  24.45% * residual
           - g1 (bem):  27.68% * residual
           - g2 (est):  26.26% * residual
           - g3 (gle):  21.61% * residual
        """
        if self.W_equiv_pretrained is None:
            logger.warning("No W_equiv_pretrained found, using noise initialization")
            self._init_group_with_noise()
            return
        
        try:
            # 步骤1: 重建 W_shared = Ain.T @ Aout.T
            Ain = self.shared_ffn2[0].weight.data.T  # (1024, r_shared)
            Aout = self.shared_ffn2[3].weight.data.T  # (r_shared, 1024)
            W_shared_reconstructed = Ain @ Aout  # (1024, 1024)
            
            logger.info(f"Reconstructing W_shared from Ain and Aout")
            logger.info(f"  Ain shape: {Ain.shape}, Aout shape: {Aout.shape}")
            logger.info(f"  W_shared_reconstructed shape: {W_shared_reconstructed.shape}")
            
            # Step 2: 计算残差
            residual = self.W_equiv_pretrained - W_shared_reconstructed  # (1024, 1024)
            logger.info(f"Step 2: Computing residual = W_equiv - W_shared_equiv")
            logger.info(f"  W_equiv shape: {self.W_equiv_pretrained.shape}")
            logger.info(f"  residual shape: {residual.shape}")
            logger.info(f"  residual magnitude: {residual.abs().mean():.6f}")
            
            # 步骤3: 按能量比例分配残差
            # 转到 CPU 和 float32 进行 SVD
            residual_fp32 = residual.to(dtype=torch.float32, device='cpu')
            
            if self.num_groups == 2:
                # 2-Group策略：根据能量占比分配残差
                logger.info(f"2-Group residual distribution:")
                logger.info(f"  g1 (bem): {self.g2_energy_ratio:.4f} * residual")
                logger.info(f"  g2 (other): {1 - self.g2_energy_ratio:.4f} * residual")
                
                # 分配残差
                W_g1 = self.g2_energy_ratio * residual_fp32  # bem
                W_g2 = (1 - self.g2_energy_ratio) * residual_fp32  # others
                
                # 步骤4: 对每个 group 的残差做低秩分解
                # Group1 (bem)
                logger.info(f"Decomposing W_g1 (bem) with r_group1={self.r_group1}")
                U_g1, S_g1, Vt_g1 = torch.linalg.svd(W_g1, full_matrices=False)
                r_eff_g1 = min(self.r_group1, len(S_g1))
                
                Bin_g1 = U_g1[:, :r_eff_g1] @ torch.diag(torch.sqrt(S_g1[:r_eff_g1]))  # (1024, r_eff_g1)
                Bout_g1 = torch.diag(torch.sqrt(S_g1[:r_eff_g1])) @ Vt_g1[:r_eff_g1, :]  # (r_eff_g1, 1024)
                
                # 扩展并赋值
                Bin_g1_expanded = torch.zeros(self.r_group1, 1024, dtype=Bin_g1.dtype, device=Bin_g1.device)
                Bin_g1_expanded[:r_eff_g1, :] = Bin_g1.T
                
                Bout_g1_expanded = torch.zeros(1024, self.r_group1, dtype=Bout_g1.dtype, device=Bout_g1.device)
                Bout_g1_expanded[:, :r_eff_g1] = Bout_g1.T
                
                self.group1_ffn2[0].weight.data.copy_(
                    Bin_g1_expanded.to(dtype=self.pretrained_w1.dtype, device=self.pretrained_w1.device)
                )
                self.group1_ffn2[3].weight.data.copy_(
                    Bout_g1_expanded.to(dtype=self.pretrained_w2.dtype, device=self.pretrained_w2.device)
                )
                
                # Group2 (others)
                logger.info(f"Decomposing W_g2 (others) with r_group2={self.r_group2}")
                U_g2, S_g2, Vt_g2 = torch.linalg.svd(W_g2, full_matrices=False)
                r_eff_g2 = min(self.r_group2, len(S_g2))
                
                Bin_g2 = U_g2[:, :r_eff_g2] @ torch.diag(torch.sqrt(S_g2[:r_eff_g2]))  # (1024, r_eff_g2)
                Bout_g2 = torch.diag(torch.sqrt(S_g2[:r_eff_g2])) @ Vt_g2[:r_eff_g2, :]  # (r_eff_g2, 1024)
                
                # 扩展并赋值
                Bin_g2_expanded = torch.zeros(self.r_group2, 1024, dtype=Bin_g2.dtype, device=Bin_g2.device)
                Bin_g2_expanded[:r_eff_g2, :] = Bin_g2.T
                
                Bout_g2_expanded = torch.zeros(1024, self.r_group2, dtype=Bout_g2.dtype, device=Bout_g2.device)
                Bout_g2_expanded[:, :r_eff_g2] = Bout_g2.T
                
                self.group2_ffn2[0].weight.data.copy_(
                    Bin_g2_expanded.to(dtype=self.pretrained_w1.dtype, device=self.pretrained_w1.device)
                )
                self.group2_ffn2[3].weight.data.copy_(
                    Bout_g2_expanded.to(dtype=self.pretrained_w2.dtype, device=self.pretrained_w2.device)
                )
                
                logger.info(f"✓ Group FFN2s initialized from unified residual (2-Group mode)")
                logger.info(f"  - g1 (bem): {100*self.g2_energy_ratio:.2f}% (rank={self.r_group1})")
                logger.info(f"  - g2 (others): {100*(1-self.g2_energy_ratio):.2f}% (rank={self.r_group2})")
                # ===== [DIAG] 验证残差分解质量 =====
                W_g1_recon = (self.group1_ffn2[0].weight.data.float().T) @ (self.group1_ffn2[3].weight.data.float().T)
                W_g2_recon = (self.group2_ffn2[0].weight.data.float().T) @ (self.group2_ffn2[3].weight.data.float().T)
                g1_err = (W_g1.to(W_g1_recon.device) - W_g1_recon).norm().item()
                g2_err = (W_g2.to(W_g2_recon.device) - W_g2_recon).norm().item()
                logger.info(f"  [DIAG] g1 recon error: {g1_err:.4f} (target ||W_g1||={W_g1.norm().item():.4f})")
                logger.info(f"  [DIAG] g2 recon error: {g2_err:.4f} (target ||W_g2||={W_g2.norm().item():.4f})")
                # 完整重建: W_shared + W_g1 + W_g2 vs W_equiv
                W_shared_r = self.shared_ffn2[0].weight.data.float().T @ self.shared_ffn2[3].weight.data.float().T
                W_total_recon = W_shared_r.to(self.W_equiv_pretrained.device) + W_g1_recon.to(self.W_equiv_pretrained.device) + W_g2_recon.to(self.W_equiv_pretrained.device)
                total_err = (self.W_equiv_pretrained.float() - W_total_recon).norm().item()
                total_norm = self.W_equiv_pretrained.float().norm().item()
                logger.info(f"  [DIAG] Total recon error ||W_equiv - (W_shared+W_g1+W_g2)||={total_err:.4f} ({100*total_err/max(total_norm,1e-8):.2f}% of ||W_equiv||)")
                # ===== [DIAG END] =====
            
            # 4-Group策略：按能量比例分配残差
            elif self.num_groups == 4:
                logger.info(f"4-Group residual distribution:")
                logger.info(f"  g0 (aeb):  {ENERGY_RATIOS_4['g0']:.4f} * residual")
                logger.info(f"  g1 (bem):  {ENERGY_RATIOS_4['g1']:.4f} * residual")
                logger.info(f"  g2 (est):  {ENERGY_RATIOS_4['g2']:.4f} * residual")
                logger.info(f"  g3 (gle):  {ENERGY_RATIOS_4['g3']:.4f} * residual")
                
                # 为每个 group 分配残差并做低秩分解
                for group_idx, (group_name, energy_ratio) in enumerate(ENERGY_RATIOS_4.items()):
                    group_ffn2 = self.group_ffn2_list[group_idx]
                    
                    # 分配残差
                    W_g = energy_ratio * residual_fp32  # (1024, 1024)
                    
                    # 低秩分解
                    logger.info(f"Decomposing W_{group_name} with r_group_4={self.r_group_4}")
                    U_g, S_g, Vt_g = torch.linalg.svd(W_g, full_matrices=False)
                    r_eff_g = min(self.r_group_4, len(S_g))
                    
                    Bin_g = U_g[:, :r_eff_g] @ torch.diag(torch.sqrt(S_g[:r_eff_g]))  # (1024, r_eff_g)
                    Bout_g = torch.diag(torch.sqrt(S_g[:r_eff_g])) @ Vt_g[:r_eff_g, :]  # (r_eff_g, 1024)
                    
                    # 扩展并赋值
                    Bin_g_expanded = torch.zeros(self.r_group_4, 1024, dtype=Bin_g.dtype, device=Bin_g.device)
                    Bin_g_expanded[:r_eff_g, :] = Bin_g.T
                    
                    Bout_g_expanded = torch.zeros(1024, self.r_group_4, dtype=Bout_g.dtype, device=Bout_g.device)
                    Bout_g_expanded[:, :r_eff_g] = Bout_g.T
                    
                    group_ffn2[0].weight.data.copy_(
                        Bin_g_expanded.to(dtype=self.pretrained_w1.dtype, device=self.pretrained_w1.device)
                    )
                    group_ffn2[3].weight.data.copy_(
                        Bout_g_expanded.to(dtype=self.pretrained_w2.dtype, device=self.pretrained_w2.device)
                    )
                
                logger.info(f"✓ Group FFN2s initialized from unified residual (4-Group mode)")
                logger.info(f"  - g0 (aeb):  24.45% (rank={self.r_group_4})")
                logger.info(f"  - g1 (bem):  27.68% (rank={self.r_group_4})")
                logger.info(f"  - g2 (est):  26.26% (rank={self.r_group_4})")
                logger.info(f"  - g3 (gle):  21.61% (rank={self.r_group_4})")
                logger.info(f"  - Residual magnitude: {residual.abs().mean():.6f}")
        
        except Exception as e:
            logger.warning(f"Failed to initialize from residual: {e}")
            logger.warning("Falling back to noise initialization")
            self._init_group_with_noise()
    
    def _init_group_with_noise(self) -> None:
        """初始化group FFN为小随机噪声"""
        for layer in self.group_ffn2_list:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=self.noise_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def _init_group_with_kaiming(self) -> None:
        """
        使用Kaiming初始化group FFN参数（标准随机初始化）
        
        适用于ReLU/SiLU激活函数
        W_shape = (out_features, in_features)
        fan_in = in_features
        std = sqrt(2 / fan_in)
        """
        logger.info(f"Initializing group FFN2 with Kaiming (random standard initialization, {self.num_groups}-group)")
        
        for idx, layer in enumerate(self.group_ffn2_list):
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        if self.num_groups == 2:
            logger.info(f"✓ Group FFN2 initialized with Kaiming (SiLU-optimized random init, 2-group)")
            logger.info(f"  - g1 (bem) rank: {self.r_group1}")
            logger.info(f"  - g2 (other) rank: {self.r_group2}")
        else:
            logger.info(f"✓ Group FFN2 initialized with Kaiming (SiLU-optimized random init, 4-group)")
            logger.info(f"  - g0 (aeb) rank: {self.r_group_4}")
            logger.info(f"  - g1 (bem) rank: {self.r_group_4}")
            logger.info(f"  - g2 (est) rank: {self.r_group_4}")
            logger.info(f"  - g3 (gle) rank: {self.r_group_4}")
    
    def forward(self, x: torch.Tensor, group: Optional[str] = None, langs: Optional[list] = None) -> torch.Tensor:
        """
        前向传播 - 支持per-batch和per-sample两种routing模式，以及2/4-group方案
        
        Per-Batch Mode（向后兼容，默认）:
            x: (batch_size, seq_len, model_dim)
            group: 'g1' or 'g2' (2-group) or 'g0','g1','g2','g3' (4-group)
            langs: None
        
        Per-Sample Mode（新增功能）:
            x: (batch_size, seq_len, model_dim)
            group: None (ignored)
            langs: ['bem', 'aeb', 'est', ...] - 每个sample有自己的语言
        
        Returns:
            输出特征 (batch_size, seq_len, model_dim)
        """
        # Shared部分 - 所有样本共享
        shared_out = self.shared_ffn2(x)
        
        # ===== Per-Sample Routing =====
        if langs is not None and len(langs) > 0:
            output = shared_out.clone()
            
            if self.num_groups == 2:
                # 2-Group per-sample routing: bem vs others
                mask_bem = torch.tensor(
                    [lang == "bem" for lang in langs],
                    dtype=torch.bool,
                    device=x.device
                )
                mask_other = ~mask_bem
                
                num_bem = mask_bem.sum().item()
                num_other = mask_other.sum().item()
                self._route_stats["bem"] = num_bem
                self._route_stats["other"] = num_other
                
                if num_bem > 0:
                    x_bem = x[mask_bem]
                    private_bem_out = self.group1_ffn2(x_bem)
                    output[mask_bem] = output[mask_bem] + private_bem_out
                
                if num_other > 0:
                    x_other = x[mask_other]
                    private_other_out = self.group2_ffn2(x_other)
                    output[mask_other] = output[mask_other] + private_other_out
            
            elif self.num_groups == 4:
                # 4-Group per-sample routing: aeb, bem, est, gle each separate
                lang_to_idx = {"aeb": 0, "bem": 1, "est": 2, "gle": 3}
                
                # 为每个语言创建mask并处理
                for lang_name, group_idx in lang_to_idx.items():
                    mask_lang = torch.tensor(
                        [lang == lang_name for lang in langs],
                        dtype=torch.bool,
                        device=x.device
                    )
                    num_lang = mask_lang.sum().item()
                    
                    if lang_name in self._route_stats:
                        self._route_stats[lang_name] = num_lang
                    else:
                        self._route_stats[lang_name] = num_lang
                    
                    if num_lang > 0:
                        x_lang = x[mask_lang]
                        private_lang_out = self.group_ffn2_list[group_idx](x_lang)
                        output[mask_lang] = output[mask_lang] + private_lang_out
            
            return output
        
        # ===== Per-Batch Routing（向后兼容） =====
        if group is None:
            group = get_current_group()
        
        # Group-specific部分
        if self.num_groups == 2:
            if group == "g1" or group == "bem":
                group_out = self.group1_ffn2(x)
            elif group == "g2" or group == "other":
                group_out = self.group2_ffn2(x)
            else:
                logger.warning(f"Unknown group '{group}' in 2-group mode, using g1")
                group_out = self.group1_ffn2(x)
        
        elif self.num_groups == 4:
            group_idx_map = {"g0": 0, "g1": 1, "g2": 2, "g3": 3, 
                            "aeb": 0, "bem": 1, "est": 2, "gle": 3}
            group_idx = group_idx_map.get(group, 0)
            group_out = self.group_ffn2_list[group_idx](x)
        
        else:
            logger.warning(f"Unknown num_groups {self.num_groups}, using group 0")
            group_out = self.group_ffn2_list[0](x)
        
        return shared_out + group_out


def apply_b1_to_model(
    model: nn.Module,
    share_ratio: float = 0.5,  # 新增：共享比例，控制 r_shared 和 k_shared
    dropout_rate: float = 0.0,
    init_strategy: str = 'residual',
    num_groups: int = 2,  # group数量 (2 或 4)
    group_assignments: Optional[Dict[str, str]] = None,  # 语言分组配置
    group_energy_ratios: Optional[Dict[str, float]] = None,  # 能量比例配置
    use_noise_init: bool = False,  # 扩展部分是否使用小噪声初始化
    target_module: str = 'encoder_ffn2',  # 新增：目标模块选择
) -> nn.Module:
    """
    将B1架构应用到模型上（支持多种目标模块选择）
    
    Args:
        model: 原始模型（SeamlessM4T或包装类）
        share_ratio: 共享比例 (默认 0.5)，控制：
                    - k_shared = int(1024 * share_ratio)  # SVD奇异值数量
                    - r_shared = int(4096 * share_ratio)  # 架构中间维度
        dropout_rate: Dropout比率（默认0.0，禁用）
        init_strategy: 初始化策略 ('residual' 或 'random'，默认 'residual')
        num_groups: 分组数量 (2 或 4，默认 2)
        group_assignments: 语言分组配置，如 {"aeb": "g1", "est": "g1", "gle": "g1", "bem": "g2"}
        group_energy_ratios: 能量比例配置，如 {"g2": 0.2768} (2-group) 或 {"g0": 0.24, ...} (4-group)
        use_noise_init: 扩展部分是否使用小噪声初始化（默认False，补0）
        target_module: 目标模块选择 ('encoder_ffn1' | 'encoder_ffn2' | 'adapter_ffn' | 'intermediate_ffn')，默认 'encoder_ffn2'
                      - 'encoder_ffn1': speech_encoder.encoder.layers[11].ffn1 (新增)
                      - 'encoder_ffn2': speech_encoder.encoder.layers[11].ffn2 (原始行为)
                      - 'adapter_ffn': speech_encoder.adapter.layers[0].ffn
                      - 'intermediate_ffn': speech_encoder.intermediate_ffn
    
    Returns:
        应用B1后的模型
    """
    # 应用动态配置
    if group_assignments is not None:
        update_language_grouping(group_assignments, num_groups)
    
    if group_energy_ratios is not None:
        logger.info(f"Using custom group energy ratios: {group_energy_ratios}")
    
    logger.info("=== Applying B1 Architecture ===")
    logger.info(f"Target module: {target_module}")
    logger.info(f"Share ratio: {share_ratio}")
    logger.info(f"Number of groups: {num_groups}")
    logger.info(f"Dropout rate: {dropout_rate}")
    logger.info(f"Initialization strategy: {init_strategy}")
    logger.info(f"Use noise init: {use_noise_init}")
    
    # 获取 speech_encoder
    if hasattr(model, 'speech_encoder'):
        speech_encoder = model.speech_encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'speech_encoder'):
        speech_encoder = model.model.speech_encoder
    else:
        raise ValueError("Cannot find speech_encoder in model")
    
    # 根据target_module选择目标模块和路径
    if target_module == 'encoder_ffn2':
        # 原始逻辑：替换encoder.layers[11].ffn2
        target_description = "encoder.layers[11].ffn2 (original behavior)"
        module_path, original_ffn, ffn_attr_name = _get_encoder_ffn2_module(speech_encoder)
    elif target_module == 'encoder_ffn1':
        # 新增：替换encoder.layers[11].ffn1
        target_description = "encoder.layers[11].ffn1"
        module_path, original_ffn, ffn_attr_name = _get_encoder_ffn1_module(speech_encoder)
    elif target_module == 'encoder_layer10_ffn2':
        # 新增：替换encoder.layers[10].ffn2
        target_description = "encoder.layers[10].ffn2 (layer 10)"
        module_path, original_ffn, ffn_attr_name = _get_encoder_ffn2_layer10_module(speech_encoder)
    elif target_module == 'adapter_ffn':
        # 新增：替换adapter.layers[0].ffn
        target_description = "speech_encoder.adapter.layers[0].ffn"
        module_path, original_ffn, ffn_attr_name = _get_adapter_ffn_module(speech_encoder)
    elif target_module == 'intermediate_ffn':
        # 新增：替换intermediate_ffn
        target_description = "speech_encoder.intermediate_ffn"
        module_path, original_ffn, ffn_attr_name = _get_intermediate_ffn_module(speech_encoder)
    else:
        raise ValueError(f"Unknown target_module: {target_module}. "
                        f"Supported: encoder_ffn1, encoder_ffn2, encoder_layer10_ffn2, adapter_ffn, intermediate_ffn")
    
    logger.info(f"✓ Target module: {target_description}")
    logger.info(f"✓ Found original FFN at: {module_path}")
    
    # 提取模型维度
    model_dim = 1024
    # 对于不同模块可能需要不同的维度检测逻辑，但SeamlessM4T都是1024
    logger.info(f"Model dimension: {model_dim}")
    
    # 创建GroupSpecificFFN2
    logger.info("Creating GroupSpecificFFN2")
    
    new_ffn2 = GroupSpecificFFN2(
        model_dim=model_dim,
        hidden_dim_original=4096,
        share_ratio=share_ratio,
        pretrained_ffn2=original_ffn,
        dropout_rate=dropout_rate,
        init_strategy=init_strategy,
        num_groups=num_groups,
        group_energy_ratios=group_energy_ratios,
        use_noise_init=use_noise_init,
    )
    
    # 移到相同设备和数据类型
    try:
        original_params = list(original_ffn.parameters())
        if original_params:
            original_param = next(iter(original_params))
            new_ffn2 = new_ffn2.to(device=original_param.device, dtype=original_param.dtype)
    except Exception as e:
        logger.warning(f"Could not move new_ffn2 to match original_ffn: {e}")
    
    # 替换模块 - 根据target_module选择不同的替换逻辑
    if target_module in ['encoder_ffn1', 'encoder_ffn2', 'encoder_layer10_ffn2']:
        # encoder层逻辑：替换指定层中的FFN
        # 重新获取layers
        if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
            layers = speech_encoder.encoder.layers
        elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
            layers = speech_encoder.inner.layers
        elif hasattr(speech_encoder, 'layers'):
            layers = speech_encoder.layers
        
        # 确定目标层
        if target_module == 'encoder_layer10_ffn2':
            layer_idx = 10
            layer = deepcopy(layers[10])
        else:  # encoder_ffn1 or encoder_ffn2
            layer_idx = 11
            layer = deepcopy(layers[11])
        
        setattr(layer, ffn_attr_name, new_ffn2)
        layers[layer_idx] = layer
        logger.info(f"✓ Replaced {ffn_attr_name} in layer {layer_idx} and put back to layers[{layer_idx}]")
        # ===== [DIAG] 验证替换真的生效 =====
        replaced = getattr(layers[layer_idx], ffn_attr_name)
        if isinstance(replaced, GroupSpecificFFN2):
            logger.info(f"  [DIAG] ✓ Confirmed: layers[{layer_idx}].{ffn_attr_name} is now GroupSpecificFFN2")
            logger.info(f"  [DIAG]   r_shared={replaced.r_shared}, num_groups={replaced.num_groups}")
            logger.info(f"  [DIAG]   shared_ffn2[0].weight norm: {replaced.shared_ffn2[0].weight.data.norm().item():.4f}")
        else:
            logger.error(f"  [DIAG] ✗ Replacement FAILED! layers[{layer_idx}].{ffn_attr_name} is still {type(replaced).__name__}")
        # 同时验证相邻层没有被意外影响
        other_idx = 10 if layer_idx == 11 else 11
        if other_idx < len(layers):
            other_ffn = getattr(layers[other_idx], ffn_attr_name, None)
            if other_ffn is not None:
                if isinstance(other_ffn, GroupSpecificFFN2):
                    logger.warning(f"  [DIAG] ⚠️  layers[{other_idx}].{ffn_attr_name} is ALSO GroupSpecificFFN2 (unexpected side-effect?)")
                else:
                    logger.info(f"  [DIAG] ✓ layers[{other_idx}].{ffn_attr_name} intact: {type(other_ffn).__name__}")
        # ===== [DIAG END] =====
        
    elif target_module == 'adapter_ffn':
        # 根据实际找到的adapter路径进行替换
        if 'speech_encoder.adaptor_layers[0]' in module_path:
            # SeamlessM4T的adaptor_layers直接是层列表
            setattr(speech_encoder.adaptor_layers[0], ffn_attr_name, new_ffn2)
        elif 'speech_encoder.adapter.layers[0]' in module_path:
            setattr(speech_encoder.adapter.layers[0], ffn_attr_name, new_ffn2)
        elif 'speech_encoder.adaptor.layers[0]' in module_path:
            setattr(speech_encoder.adaptor.layers[0], ffn_attr_name, new_ffn2)
        elif 'speech_encoder.inner.adapter.layers[0]' in module_path:
            setattr(speech_encoder.inner.adapter.layers[0], ffn_attr_name, new_ffn2)
        elif 'speech_encoder.inner.adaptor.layers[0]' in module_path:
            setattr(speech_encoder.inner.adaptor.layers[0], ffn_attr_name, new_ffn2)
        elif 'speech_encoder.inner.adaptor_layers[0]' in module_path:
            setattr(speech_encoder.inner.adaptor_layers[0], ffn_attr_name, new_ffn2)
        else:
            # 通用的动态替换逻辑
            logger.warning(f"Unknown adapter path: {module_path}, trying dynamic replacement")
            # 从module_path解析出需要的对象路径
            if 'adaptor_layers[0]' in module_path:
                if hasattr(speech_encoder, 'adaptor_layers') and len(speech_encoder.adaptor_layers) > 0:
                    setattr(speech_encoder.adaptor_layers[0], ffn_attr_name, new_ffn2)
                elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'adaptor_layers'):
                    setattr(speech_encoder.inner.adaptor_layers[0], ffn_attr_name, new_ffn2)
        logger.info(f"✓ Replaced {module_path}")
        
    elif target_module == 'intermediate_ffn':
        # 直接替换intermediate_ffn - 支持多种位置
        if 'speech_encoder.intermediate_ffn' in module_path:
            setattr(speech_encoder, ffn_attr_name, new_ffn2)
        elif 'speech_encoder.inner.intermediate_ffn' in module_path:
            setattr(speech_encoder.inner, ffn_attr_name, new_ffn2)
        logger.info(f"✓ Replaced {module_path}")
    
    logger.info("✓ B1 architecture applied successfully!")
    logger.info(f"  - Modified: {target_description}")
    logger.info(f"  - Share ratio: {share_ratio}")
    r_shared = int(4096 * share_ratio)
    if num_groups == 2:
        r_group = (4096 - r_shared) // 2
        logger.info(f"  - Architecture: W ≈ W_shared(r={r_shared}) + W_g1(r={r_group}, bem) + W_g2(r={r_group}, others)")
    else:
        r_group_4 = (4096 - r_shared) // 4
        logger.info(f"  - Architecture: W ≈ W_shared(r={r_shared}) + W_g0(r={r_group_4}, aeb) + W_g1(r={r_group_4}, bem) + W_g2(r={r_group_4}, est) + W_g3(r={r_group_4}, gle)")
    
    return model


def freeze_model_for_b1(model: nn.Module, target_module: str = 'encoder_ffn2') -> None:
    """
    冻结模型参数，根据target_module选择不同的解冻策略
    
    训练参数范围说明：
    - encoder_ffn1/ffn2: 解冻speech_encoder.inner.layers[10-11] (encoder的最后两层)
                         + GroupSpecificFFN2 B1架构参数
    - adapter_ffn: ONLY解冻speech_encoder.adaptor_layers[0] (单个adapter层)
                   + GroupSpecificFFN2 B1架构参数
                   注意：encoder layers[10-11]保持冻结！
    
    关键区别：
    - encoder_ffn1/ffn2：训练encoder layers + B1参数（标准方案）
    - adapter_ffn：只训练adapter层 + B1参数（不训练encoder layers）
    
    Args:
        model: 应用B1后的模型
        target_module: 目标模块类型，决定解冻哪些参数
                      - 'encoder_ffn1': 解冻layers 10和11 + B1参数
                      - 'encoder_ffn2': 解冻layers 10和11 + B1参数（原始行为）
                      - 'adapter_ffn': 只解冻adaptor_layers[0] + B1参数（不训练encoder）
    """
    logger.info("=== Freezing Model for B1 Training ===")
    logger.info(f"Target module: {target_module}")
    
    # 1. 冻结所有参数
    logger.info("Step 1: Freezing all parameters")
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 获取 speech_encoder
    if hasattr(model, 'speech_encoder'):
        speech_encoder = model.speech_encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'speech_encoder'):
        speech_encoder = model.model.speech_encoder
    else:
        raise ValueError("Cannot find speech_encoder in model")
    
    # 3. 根据target_module选择解冻策略
    if target_module in ['encoder_ffn1', 'encoder_ffn2']:
        # encoder第11层逻辑：解冻第10和11层
        layers = None
        
        if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
            layers = speech_encoder.encoder.layers
        elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
            layers = speech_encoder.inner.layers
        elif hasattr(speech_encoder, 'layers'):
            layers = speech_encoder.layers
        else:
            raise ValueError("Cannot find encoder.layers in speech_encoder")
        
        logger.info("Step 2: Unfreezing layers 10 and 11")
        for layer_idx in [10, 11]:
            logger.info(f"  Unfreezing layer {layer_idx}")
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
                
    elif target_module == 'encoder_layer10_ffn2':
        # encoder第10层ffn2逻辑：解冻第10和11层（与encoder_ffn2一致，提供足够的trainable capacity）
        layers = None
        
        if hasattr(speech_encoder, 'encoder') and hasattr(speech_encoder.encoder, 'layers'):
            layers = speech_encoder.encoder.layers
        elif hasattr(speech_encoder, 'inner') and hasattr(speech_encoder.inner, 'layers'):
            layers = speech_encoder.inner.layers
        elif hasattr(speech_encoder, 'layers'):
            layers = speech_encoder.layers
        else:
            raise ValueError("Cannot find encoder.layers in speech_encoder")
        
        logger.info("Step 2: Unfreezing layers 10 and 11 (layer10.ffn2 replaced + layer11 as context)")
        for layer_idx in [10, 11]:
            logger.info(f"  Unfreezing layer {layer_idx}")
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
                
    elif target_module == 'adapter_ffn':
        # 只解冻adapter相关参数（不解冻encoder layers 10-11）
        # 注意：adapter layer 0 约 46M 参数，与 layers 10+11 (~48M) 相当
        logger.info("Step 2: Unfreezing adapter.layers[0] parameters ONLY (encoder layers remain frozen)")
        unfrozen_count = 0
        
        # 优先：speech_encoder.adapter.layers[0]（HuggingFace SeamlessM4TConformerAdapter 结构）
        if hasattr(speech_encoder, 'adapter') and hasattr(speech_encoder.adapter, 'layers') and len(speech_encoder.adapter.layers) > 0:
            logger.info("  Unfreezing speech_encoder.adapter.layers[0] (HuggingFace format)")
            for name, param in speech_encoder.adapter.layers[0].named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
                logger.debug(f"Unfroze: adapter.layers.0.{name}")
            # 其他 adapter 层保持冻结
            for i in range(1, len(speech_encoder.adapter.layers)):
                for param in speech_encoder.adapter.layers[i].parameters():
                    param.requires_grad = False
        # 备用：speech_encoder.adaptor_layers[0]（fairseq2 格式）
        elif hasattr(speech_encoder, 'adaptor_layers') and len(speech_encoder.adaptor_layers) > 0:
            logger.info("  Unfreezing speech_encoder.adaptor_layers[0] (fairseq2 format)")
            for name, param in speech_encoder.adaptor_layers[0].named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
                logger.debug(f"Unfroze: adaptor_layers.0.{name}")
            for i in range(1, len(speech_encoder.adaptor_layers)):
                for param in speech_encoder.adaptor_layers[i].parameters():
                    param.requires_grad = False
        else:
            # 最后备用：按 name 匹配（支持 adapter.layers.0 和 adaptor_layers.0 两种路径）
            logger.warning("  Falling back to name-based parameter search")
            for name, param in model.named_parameters():
                if 'adapter.layers.0' in name or 'adaptor_layers.0' in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    logger.debug(f"Unfroze: {name}")
        
        logger.info(f"Unfrozen {unfrozen_count} adapter-specific parameter tensors")
        logger.info("IMPORTANT: encoder layers[10-11] remain FROZEN for adapter_ffn target")
        logger.info("           B1 training focuses only on adapter layer (~46M params).")
            
    elif target_module == 'intermediate_ffn':
        # 新增：只解冻intermediate_ffn参数
        logger.info("Step 2: Unfreezing intermediate_ffn parameters")
        if hasattr(speech_encoder, 'intermediate_ffn'):
            logger.info("  Unfreezing speech_encoder.intermediate_ffn")
            for param in speech_encoder.intermediate_ffn.parameters():
                param.requires_grad = True
        else:
            logger.warning("speech_encoder does not have intermediate_ffn")
    
    else:
        raise ValueError(f"Unknown target_module: {target_module}")
        
    # 4. 验证冻结结果
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info("\n=== Freeze Verification ===")
    logger.info(f"Target module: {target_module}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    # 5. 显示可训练参数的详细信息
    logger.info("\n=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}: {param.numel():,}")
