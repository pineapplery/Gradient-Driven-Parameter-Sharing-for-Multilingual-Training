#!/usr/bin/env python3
"""
B1架构微调脚本 - 使用官方trainer框架

关键特点：
1. 使用官方的 UnitYFinetune + UnitYDataLoaderWithFbank + BatchingConfig
2. 应用B1架构：只替换encoder第11层的ffn2
3. 保持和基础架构完全一致的训练主循环
4. 减少自定义代码，降低报错风险

B1架构说明：
- 加性分解：W ≈ W_shared + W_group_g
- 只修改第11层的ffn2，其他层保持预训练权重
- 语言组：g1 (aeb, est, gle), g2 (bem)
- 训练第10和11层（其他层冻结）
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from seamless_communication.cli.m4t.finetune import dist_utils
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

# 导入B1最小实现
# 当前文件在 scripts/ 目录，b1_minimal.py 在 ../models/seamless_m4t_medium/
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "seamless_m4t_medium"))
from b1_minimal import (
    apply_b1_to_model,
    freeze_model_for_b1,
    set_current_group,
    get_group_for_lang,
)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune_b1_trainer")


def freeze_shared_only_for_b1(model):
    """
    两阶段训练专用：只冻结shared_ffn2参数，解冻所有group_ffn2参数
    
    用于第一阶段训练策略：加载unified checkpoint后，冻结shared，只训练group FFN2
    
    Args:
        model: 应用B1后的模型
    """
    logger.info("=== Freezing Shared Parameters for Two-Stage Training ===")
    
    # 1. 冻结所有参数
    logger.info("Step 1: Freezing all parameters")
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 只解冻带有param_group='group'标记的参数
    logger.info("Step 2: Unfreezing group-specific parameters only")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if hasattr(param, 'param_group') and param.param_group == 'group':
            param.requires_grad = True
            unfrozen_count += 1
            logger.info(f"  Unfrozen: {name}")
    
    # 3. 验证冻结结果
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info("\n=== Freeze Verification (Two-Stage Mode) ===")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters (group only): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Frozen parameters (shared + others): {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    logger.info(f"Number of unfrozen parameter tensors: {unfrozen_count}")


class PCGradHandler:
    """
    PCGrad (Projected Conflicting Gradient) Handler
    
    用于检测和解决 group1 和 group2 之间的梯度冲突。
    
    核心思想：
    - 如果两个group的梯度方向冲突（cosine < 0），则将g2的梯度投影到g1的正交空间
    - 投影公式：g2_new = g2 - (g2·g1 / ||g1||²) * g1
    
    Args:
        model: B1模型
        log_steps: 每N步输出一次冲突统计
        logger: 日志记录器
    """
    
    def __init__(self, model, log_steps=3000, logger=None):
        self.model = model
        self.log_steps = log_steps
        self.logger = logger or logging.getLogger(__name__)
        
        # 统计数据
        self.total_steps = 0
        self.conflict_count = 0
        self.total_checks = 0
        self.cosine_sum = 0.0
        
        # 收集group1和group2的参数
        self.group1_params = []
        self.group2_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if hasattr(param, 'param_group') and param.param_group == 'group':
                # 判断是group1还是group2（通过名称）
                if 'group1_ffn2' in name:
                    self.group1_params.append(param)
                elif 'group2_ffn2' in name:
                    self.group2_params.append(param)
        
        self.logger.info(f"✓ PCGrad initialized:")
        self.logger.info(f"  - Group1 parameters: {len(self.group1_params)}")
        self.logger.info(f"  - Group2 parameters: {len(self.group2_params)}")
        self.logger.info(f"  - Conflict threshold: cosine < 0")
        self.logger.info(f"  - Log interval: every {log_steps} steps")
    
    def apply_pcgrad(self):
        """
        应用PCGrad梯度修正
        
        在每次backward()后、optimizer.step()前调用
        """
        if len(self.group1_params) == 0 or len(self.group2_params) == 0:
            return
        
        # 收集梯度
        g1_grads = []
        g2_grads = []
        
        for param in self.group1_params:
            if param.grad is not None:
                g1_grads.append(param.grad.view(-1))
        
        for param in self.group2_params:
            if param.grad is not None:
                g2_grads.append(param.grad.view(-1))
        
        if len(g1_grads) == 0 or len(g2_grads) == 0:
            return
        
        # 拼接成单个向量
        g1 = torch.cat(g1_grads)
        g2 = torch.cat(g2_grads)
        
        # 计算cosine similarity
        dot_product = torch.dot(g1, g2)
        g1_norm = torch.norm(g1)
        g2_norm = torch.norm(g2)
        
        if g1_norm > 0 and g2_norm > 0:
            cosine_sim = dot_product / (g1_norm * g2_norm)
            
            # 更新统计
            self.total_checks += 1
            self.cosine_sum += cosine_sim.item()
            
            # 检测冲突
            if cosine_sim < 0:
                self.conflict_count += 1
                
                # 应用PCGrad：将g2投影到g1的正交空间
                # g2_new = g2 - (g2·g1 / ||g1||²) * g1
                projection_coeff = dot_product / (g1_norm ** 2)
                
                # 修正g2的梯度
                g2_corrected = g2 - projection_coeff * g1
                
                # 将修正后的梯度写回参数
                offset = 0
                for param in self.group2_params:
                    if param.grad is not None:
                        numel = param.grad.numel()
                        param.grad.copy_(g2_corrected[offset:offset+numel].view_as(param.grad))
                        offset += numel
        
        # 更新步数
        self.total_steps += 1
        
        # 定期输出统计
        if self.total_steps % self.log_steps == 0 and self.total_checks > 0:
            conflict_rate = 100.0 * self.conflict_count / self.total_checks
            avg_cosine = self.cosine_sum / self.total_checks
            
            self.logger.info(f"\n=== PCGrad Statistics (Step {self.total_steps}) ===")
            self.logger.info(f"  Total gradient checks: {self.total_checks}")
            self.logger.info(f"  Conflicts detected: {self.conflict_count} ({conflict_rate:.1f}%)")
            self.logger.info(f"  Average cosine similarity: {avg_cosine:.4f}")
            
            if conflict_rate > 50:
                self.logger.warning("  ⚠️  High conflict rate (>50%)! Consider adjusting learning rates or architecture.")
            elif conflict_rate > 0:
                self.logger.info(f"  ✓ Gradient conflicts resolved via PCGrad projection")
            else:
                self.logger.info("  ✓ No conflicts detected - gradients aligned")


def setup_grouped_learning_rate(model, finetune, shared_lr, group_lr, logger):
    """
    为shared_ffn2和group_ffn2设置不同的学习率
    
    Args:
        model: B1模型
        finetune: UnitYFinetune实例
        shared_lr: shared_ffn2的学习率
        group_lr: group_ffn2的学习率
        logger: 日志记录器
    """
    try:
        # 获取optimizer
        optimizer = finetune.optimizer
        if optimizer is None:
            logger.warning("Optimizer not yet created, cannot set grouped learning rates")
            return
        
        # 分类参数
        shared_params = []
        group_params = []
        other_params = []
        
        # 遍历所有参数，根据标记分类
        for param in model.parameters():
            if not param.requires_grad:
                continue
            
            if hasattr(param, 'param_group'):
                if param.param_group == 'shared':
                    shared_params.append(param)
                elif param.param_group == 'group':
                    group_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)
        
        logger.info(f"Found {len(shared_params)} shared parameters")
        logger.info(f"Found {len(group_params)} group parameters")
        logger.info(f"Found {len(other_params)} other parameters")
        
        # 获取原始param_group的所有超参数（为了保留AdamW所需的所有配置）
        original_group = optimizer.param_groups[0]
        base_group_config = {k: v for k, v in original_group.items() if k != 'params' and k != 'lr'}
        
        logger.info(f"Base optimizer config: {base_group_config}")
        
        # 重新构造param_groups - 保留所有原始超参数
        new_param_groups = []
        
        if shared_params:
            group_dict = {'params': shared_params, 'lr': shared_lr, 'name': 'shared_ffn2'}
            group_dict.update(base_group_config)  # 添加所有AdamW超参数
            new_param_groups.append(group_dict)
            logger.info(f"✓ Shared parameters: lr = {shared_lr}")
        
        if group_params:
            group_dict = {'params': group_params, 'lr': group_lr, 'name': 'group_ffn2'}
            group_dict.update(base_group_config)  # 添加所有AdamW超参数
            new_param_groups.append(group_dict)
            logger.info(f"✓ Group parameters: lr = {group_lr}")
        
        if other_params:
            group_dict = {'params': other_params, 'lr': shared_lr, 'name': 'other_layers'}
            group_dict.update(base_group_config)  # 添加所有AdamW超参数
            new_param_groups.append(group_dict)
            logger.info(f"✓ Other parameters: lr = {shared_lr}")
        
        # 更新optimizer的param_groups
        optimizer.param_groups = new_param_groups
        logger.info("✓ Grouped learning rates configured successfully!")
        
    except Exception as e:
        logger.error(f"Failed to setup grouped learning rates: {e}")
        logger.warning("Continuing with single learning rate...")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="B1架构微调脚本 - 使用官方trainer框架"
    )
    parser.add_argument("--train_dataset", type=Path, required=True,
                       help="Path to train manifest")
    parser.add_argument("--eval_dataset", type=Path, required=True,
                       help="Path to eval manifest")
    parser.add_argument("--model_name", type=str, default="seamlessM4T_medium",
                       help="Model name")
    parser.add_argument("--save_model_to", type=Path, required=True,
                       help="Path to save finetuned model")
    parser.add_argument("--seed", type=int, default=2343,
                       help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--max_epochs", type=int, default=30,
                       help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--eval_steps", type=int, default=5000,
                       help="Evaluate every N steps")
    parser.add_argument("--log_steps", type=int, default=5000,
                       help="Log every N steps")
    parser.add_argument("--max_src_tokens", type=int, default=20000,
                       help="Maximum source tokens per batch")
    parser.add_argument("--update_freq", type=int, default=1,
                       help="Gradient accumulation steps (update_freq)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--use_fbank", action="store_true", default=True,
                       help="Use precomputed fbank features")
    # B1架构参数
    parser.add_argument("--share_ratio", type=float, default=0.5,
                       help="Share ratio for shared vs group parameters (default: 0.5). "
                            "Controls k_shared = int(1024 * share_ratio) and "
                            "r_shared = int(4096 * share_ratio). "
                            "Example: 0.5 → r_shared=2048, k_shared=512")
    parser.add_argument("--use_grouped_lr", action="store_true", default=False,
                       help="Enable different learning rates for shared vs group parameters")
    parser.add_argument("--group_learning_rate", type=float, default=None,
                       help="Learning rate for group-specific FFN (e.g., 3e-5 or 5e-5). If None, uses 3x of --learning_rate")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                       help="Dropout rate for FFN2 (shared and group). (default: 0.0, disabled)")
    parser.add_argument("--use_noise_init", action="store_true", default=False,
                       help="Use small noise initialization for expanded dimensions (default: False, use zero padding)")
    
    # Checkpoint加载参数（推荐）
    parser.add_argument("--load_checkpoint_path", type=Path, default=None,
                       help="Load checkpoint and train both shared and group parameters with different learning rates (recommended)")
    
    # 两阶段训练参数（deprecated，保留向后兼容）
    parser.add_argument("--two_stage_training", action="store_true", default=False,
                       help="Enable two-stage training: load unified checkpoint, freeze shared, train group only")
    parser.add_argument("--unified_checkpoint_path", type=Path, default=None,
                       help="Path to unified checkpoint (e.g., unify_checkpoint1.pt). Only used with --two_stage_training")
    parser.add_argument("--private_lr", type=float, default=None,
                       help="Learning rate for private (group) parameters in two-stage training (e.g., 5e-5 to 1e-4)")
    parser.add_argument("--private_weight_decay", type=float, default=1e-3,
                       help="Weight decay for private parameters in two-stage training (default: 1e-3)")
    
    # PCGrad梯度修正参数
    parser.add_argument("--use_pcgrad", action="store_true", default=False,
                       help="Enable PCGrad (Projected Conflicting Gradient) to resolve gradient conflicts between group1 and group2")
    parser.add_argument("--pcgrad_log_steps", type=int, default=3000,
                       help="Log PCGrad conflict statistics every N steps (default: 3000)")
    
    # 初始化策略参数
    parser.add_argument("--init_strategy", type=str, default='residual', choices=['residual', 'random'],
                       help="Initialization strategy for group FFN2 parameters: 'residual' (default, based on checkpoint) or 'random' (Kaiming initialization)")
    
    # Group数量参数
    parser.add_argument("--num_groups", type=int, default=2, choices=[2, 4],
                       help="Number of language groups (default: 2). "
                            "2 = bem vs others, 4 = aeb/bem/est/gle separate. "
                            "Group ranks are automatically calculated from share_ratio.")
    
    # 语言分组配置参数
    parser.add_argument("--group_assignments", type=str, default=None,
                       help="Language group assignments in format 'aeb:g1,est:g1,gle:g1,bem:g2'. "
                            "If not provided, uses default grouping. "
                            "For 2-group: g1 and g2. For 4-group: g0, g1, g2, g3.")
    
    # 能量比例配置参数
    parser.add_argument("--group_energy_ratios", type=str, default=None,
                       help="Group energy ratios in format 'g2:0.2768' (2-group) or 'g0:0.24,g1:0.28,g2:0.26,g3:0.22' (4-group). "
                            "Used for residual initialization. "
                            "If not provided, uses default ratios.")
    
    # 新增：目标模块选择参数
    parser.add_argument("--target_module", type=str, default='encoder_ffn2',
                       choices=['encoder_ffn1', 'encoder_ffn2', 'encoder_layer10_ffn2', 'adapter_ffn', 'intermediate_ffn'],
                       help="Target module to apply B1 architecture (default: encoder_ffn2). "
                            "Options: "
                            "encoder_ffn1 = speech_encoder.encoder.layers[11].ffn1 (layer 11), "
                            "encoder_ffn2 = speech_encoder.encoder.layers[11].ffn2 (layer 11, original), "
                            "encoder_layer10_ffn2 = speech_encoder.encoder.layers[10].ffn2 (layer 10, new), "
                            "adapter_ffn = speech_encoder.adaptor_layers[0].ffn, "
                            "intermediate_ffn = speech_encoder.intermediate_ffn")
    
    return parser


def main():
    args = init_parser().parse_args()

    # 初始化分布式环境
    dist_utils.init_distributed([logger])
    
    # 设置数据加载超时和worker相关的环境变量（防止DataLoader worker被意外kill）
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['OMP_NUM_THREADS'] = '1'  # 防止OpenMP线程竞争
    
    # 设置数据类型
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    logger.info("=== B1 Trainer-based Training Script ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Share ratio: {args.share_ratio}")
    r_shared = int(4096 * args.share_ratio)
    r_remaining = 4096 - r_shared
    logger.info(f"  r_shared: {r_shared} ({100*args.share_ratio:.1f}%)")
    logger.info(f"  r_remaining: {r_remaining} ({100*(1-args.share_ratio):.1f}%)")
    logger.info(f"Initialization strategy: {args.init_strategy}")
    logger.info(f"Train dataset: {args.train_dataset}")
    logger.info(f"Eval dataset: {args.eval_dataset}")
    logger.info(f"Use fbank: {args.use_fbank}")
    
    # 加载tokenizer
    logger.info("Loading tokenizers...")
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    # 加载模型
    logger.info(f"Loading model: {args.model_name}")
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    assert model.target_vocab_info == text_tokenizer.vocab_info
    
    # 移除不需要的模块（S2T任务）
    if model.t2u_model is not None:
        logger.info("Removing t2u_model")
        model.t2u_model = None
    if model.text_encoder is not None:
        logger.info("Removing text_encoder")
        model.text_encoder = None
    
    # 加载Checkpoint：支持两种模式
    # 1. --load_checkpoint_path：加载后同时训练shared和group（推荐）
    # 2. --two_stage_training + --unified_checkpoint_path：加载后只训练group
    if args.load_checkpoint_path or args.two_stage_training:
        if args.load_checkpoint_path:
            checkpoint_path = args.load_checkpoint_path
            training_mode = "mixed"
            logger.info("\n=== Mixed Training: Loading Checkpoint (Train Shared + Group) ===")
        else:
            checkpoint_path = args.unified_checkpoint_path
            training_mode = "two_stage"
            logger.info("\n=== Two-Stage Training: Loading Unified Checkpoint (Train Group Only) ===")
        
        if checkpoint_path is None:
            raise ValueError("Checkpoint path is required")
        
        logger.info(f"Loading from: {checkpoint_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取模型权重（支持多种checkpoint格式）
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            logger.info("Loaded from checkpoint['model']")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info("Loaded from checkpoint['state_dict']")
        else:
            state_dict = checkpoint
            logger.info("Loaded checkpoint as state_dict directly")
        
        # 移除'model.'前缀（如果有）
        clean_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            clean_state_dict[new_key] = value
        
        logger.info(f"Total checkpoint keys: {len(clean_state_dict)}")
        
        # 加载权重（strict=False以忽略missing keys如text_encoder）
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        
        logger.info(f"✓ Unified checkpoint loaded successfully")
        
        logger.info("✓ Checkpoint loaded successfully")
        
        if missing_keys:
            missing_count = len(missing_keys)
            logger.info(f"  Missing keys from checkpoint: {missing_count} (expected - removed modules)")
            # 区分哪些是被删除的模块（t2u, text_encoder）
            removed_keys = [k for k in missing_keys if 't2u_model' in k or 'text_encoder' in k]
            other_missing = [k for k in missing_keys if k not in removed_keys]
            if removed_keys:
                logger.info(f"    - Removed modules (t2u, text_encoder): {len(removed_keys)} keys")
            if other_missing:
                logger.warning(f"    - WARNING: Other missing keys: {len(other_missing)} keys")
        
        if unexpected_keys:
            logger.info(f"  Unexpected keys in checkpoint: {len(unexpected_keys)} (ignored)")
        
        logger.info("✓ Model initialized with checkpoint weights (S2TT complete)")
        logger.info("  speech_encoder, text_decoder, and adaptor layers loaded")
        logger.info(f"  Next: Apply B1 architecture, training mode={training_mode}")
        
        # ===== [DIAG] 验证checkpoint关键层权重是否真的被加载 =====
        logger.info("[DIAG] Verifying checkpoint layer weights are loaded (not random):")
        se = model.speech_encoder
        # 动态找到 layers
        _layers = None
        if hasattr(se, 'inner') and hasattr(se.inner, 'layers'):
            _layers = se.inner.layers
            logger.info("[DIAG]   Using speech_encoder.inner.layers")
        elif hasattr(se, 'encoder') and hasattr(se.encoder, 'layers'):
            _layers = se.encoder.layers
            logger.info("[DIAG]   Using speech_encoder.encoder.layers")
        
        if _layers is not None:
            kaiming_std = (2.0 / 1024) ** 0.5  # ~0.044
            for chk_idx in [10, 11]:
                layer = _layers[chk_idx]
                for ffn_name in ['ffn1', 'ffn2']:
                    ffn = getattr(layer, ffn_name, None)
                    if ffn is None:
                        continue
                    proj = getattr(ffn, 'inner_proj', None) or getattr(ffn, 'intermediate_dense', None)
                    if proj is not None and hasattr(proj, 'weight'):
                        w = proj.weight.data.float()
                        diff = abs(w.std().item() - kaiming_std)
                        status = "✓ pretrained" if diff > 0.01 else "⚠️  MAY BE RANDOM (std≈Kaiming)"
                        logger.info(f"[DIAG]   layer[{chk_idx}].{ffn_name}.inner_proj: std={w.std():.4f}  {status}")
        # ===== [DIAG END] =====
    
    # 解析语言分组配置
    group_assignments = None
    if args.group_assignments:
        try:
            group_assignments = {}
            for pair in args.group_assignments.split(','):
                lang, group = pair.strip().split(':')
                group_assignments[lang.strip()] = group.strip()
            logger.info(f"Parsed group_assignments: {group_assignments}")
        except Exception as e:
            logger.error(f"Failed to parse --group_assignments: {e}")
            logger.error(f"Expected format: 'aeb:g1,est:g1,gle:g1,bem:g2'")
            raise
    
    # 解析能量比例配置
    group_energy_ratios = None
    if args.group_energy_ratios:
        try:
            group_energy_ratios = {}
            for pair in args.group_energy_ratios.split(','):
                group, ratio = pair.strip().split(':')
                group_energy_ratios[group.strip()] = float(ratio.strip())
            logger.info(f"Parsed group_energy_ratios: {group_energy_ratios}")
        except Exception as e:
            logger.error(f"Failed to parse --group_energy_ratios: {e}")
            logger.error(f"Expected format: 'g1:0.723,g2:0.277'")
            raise
    
    # 应用B1架构
    logger.info("\n=== Applying B1 Architecture ===")
    logger.info(f"Configuration:")
    logger.info(f"  - Target module: {args.target_module}")
    logger.info(f"  - Number of groups: {args.num_groups}")
    logger.info(f"  - Initialization strategy: {args.init_strategy}")
    logger.info(f"  - Share ratio: {args.share_ratio}")
    r_shared = int(4096 * args.share_ratio)
    r_remaining = 4096 - r_shared
    logger.info(f"  - r_shared: {r_shared}")
    if args.num_groups == 2:
        r_group = r_remaining // 2
        logger.info(f"  - r_group (each): {r_group}")
    else:
        r_group_4 = r_remaining // 4
        logger.info(f"  - r_group (each): {r_group_4}")
    if group_assignments:
        logger.info(f"  - Group assignments: {group_assignments}")
    if group_energy_ratios:
        logger.info(f"  - Group energy ratios: {group_energy_ratios}")
    
    model = apply_b1_to_model(
        model=model,
        share_ratio=args.share_ratio,
        dropout_rate=args.dropout_rate,
        init_strategy=args.init_strategy,
        num_groups=args.num_groups,
        group_assignments=group_assignments,
        group_energy_ratios=group_energy_ratios,
        use_noise_init=args.use_noise_init,
        target_module=args.target_module,  # 新增参数
    )
    
    # 冻结模型参数
    logger.info("\n=== Freezing Model Parameters ===")
    if args.two_stage_training:
        # 两阶段训练：只冻结shared，解冻所有group参数
        logger.info("Mode: Two-Stage Training - Only group parameters will be trained")
        freeze_shared_only_for_b1(model)
    elif args.load_checkpoint_path:
        # 混合训练：不冻结任何参数，同时训练shared和group（但学习率不同）
        logger.info("Mode: Mixed Training - Both shared and group parameters will be trained with different learning rates")
        freeze_model_for_b1(model, target_module=args.target_module)
    else:
        # 默认策略：冻结除layer11外的所有参数
        logger.info("Mode: Default Training - Training layer 11 only")
        freeze_model_for_b1(model, target_module=args.target_module)
    
    # ===== [DIAG] 训练前完整参数状态快照 =====
    logger.info("[DIAG] ===== POST-FREEZE PARAMETER STATUS SNAPSHOT =====")
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_total    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"[DIAG]   Trainable params: {trainable_total:,}")
    logger.info(f"[DIAG]   Frozen params:    {frozen_total:,}")
    # 按模块分类汇总
    from collections import defaultdict
    module_stats = defaultdict(lambda: [0, 0])  # [trainable, frozen]
    for name, param in model.named_parameters():
        parts = name.split('.')
        # 提取第一个有意义的路径段 (e.g. speech_encoder.inner.layers.10 → "layer_10")
        key = name
        for part in ['layers.10', 'layers.11', 'adaptor_layers.0', 'shared_ffn2', 'group1_ffn2', 'group2_ffn2',
                     'group0_ffn2', 'group3_ffn2', 'text_decoder']:
            if part in name:
                key = part
                break
        if param.requires_grad:
            module_stats[key][0] += param.numel()
        else:
            module_stats[key][1] += param.numel()
    # 只输出包含可训练参数的条目
    for key, (trainable, frozen) in sorted(module_stats.items()):
        if trainable > 0:
            logger.info(f"[DIAG]   {key:45s} trainable={trainable:>10,}  frozen={frozen:>10,}")
    # 验证 GroupSpecificFFN2 是否存在且可训练
    gsffn2_trainable = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and any(x in name for x in ['shared_ffn2', 'group1_ffn2', 'group2_ffn2',
                                                         'group0_ffn2', 'group3_ffn2'])
    )
    if gsffn2_trainable == 0:
        logger.error("[DIAG] ⚠️  GroupSpecificFFN2 has ZERO trainable params! B1 private branches cannot learn!")
    else:
        logger.info(f"[DIAG]   GroupSpecificFFN2 total trainable: {gsffn2_trainable:,}")
    logger.info("[DIAG] ===================================================")
    # ===== [DIAG END] =====
    
    # 移到设备
    logger.info(f"\nMoving model to device: {args.device}")
    model = model.to(torch.device(args.device))
    
    # 创建数据加载器（使用官方DataLoader）
    logger.info("\n=== Creating DataLoaders ===")
    from dataloader_with_fbank import UnitYDataLoaderWithFbank, BatchingConfig
    
    batching_config = BatchingConfig(
        batch_size=args.batch_size,
        rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        max_audio_length_sec=12.0,  # 降低到 12 秒以避免超过 4096 序列长度限制
        float_dtype=float_dtype,
        use_fbank=args.use_fbank,
    )
    
    train_dataloader = UnitYDataLoaderWithFbank(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=batching_config,
        dataset_manifest_path=str(args.train_dataset),
        max_src_tokens_per_batch=args.max_src_tokens,
    )
    
    eval_dataloader = UnitYDataLoaderWithFbank(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=batching_config,
        dataset_manifest_path=str(args.eval_dataset),
    )
    
    logger.info(f"✓ Train dataloader created")
    logger.info(f"✓ Eval dataloader created")
    
    # 创建训练参数（使用官方trainer）
    logger.info("\n=== Creating Trainer ===")
    from seamless_communication.cli.m4t.finetune import trainer
    
    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=trainer.FinetuneMode.SPEECH_TO_TEXT,
        save_model_path=args.save_model_to,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        update_freq=args.update_freq,
    )
    
    logger.info(f"Finetune Params: {finetune_params}")
    
    # 创建trainer实例
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=None,  # 已经通过freeze_model_for_b1冻结了
    )
    
    # ========== 加载训练manifest以获取语言标签（用于per-sample routing）==========
    logger.info("\n=== Loading Training Manifest for Per-Sample Routing ===")
    import json
    
    try:
        with open(args.train_dataset, 'r') as f:
            train_manifest = [json.loads(line) for line in f]
        
        logger.info(f"✓ Loaded {len(train_manifest)} training samples")
        
        # 构建sample_idx -> lang映射
        sample_to_lang = {}
        lang_counts = {}
        for idx, sample in enumerate(train_manifest):
            lang = sample.get('source', {}).get('lang', 'aeb')
            sample_to_lang[idx] = lang
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        logger.info("Language distribution in training data:")
        for lang, count in sorted(lang_counts.items()):
            # 动态获取正确的group标签（根据num_groups参数）
            group = get_group_for_lang(lang, args.num_groups)
            percentage = 100.0 * count / len(train_manifest)
            logger.info(f"  {lang} ({group}): {count} samples ({percentage:.1f}%)")
        
        per_sample_routing_enabled = True
        
    except Exception as e:
        logger.warning(f"Failed to load manifest for per-sample routing: {e}")
        logger.warning("Falling back to per-batch routing (using thread-local group)")
        sample_to_lang = {}
        per_sample_routing_enabled = False
    
    # ========== 同样加载eval manifest，支持eval时的per-sample routing ==========
    eval_sample_to_lang = {}
    try:
        with open(args.eval_dataset, 'r') as f:
            eval_manifest_lines = [json.loads(line) for line in f]
        for idx, sample in enumerate(eval_manifest_lines):
            lang = sample.get('source', {}).get('lang', 'aeb')
            eval_sample_to_lang[idx] = lang
        logger.info(f"✓ Loaded {len(eval_manifest_lines)} eval samples for per-sample routing")
    except Exception as e:
        logger.warning(f"Failed to load eval manifest for routing: {e}, eval will use shared-only fallback")
        eval_sample_to_lang = {}
    
    # ========== Monkey-patch GroupSpecificFFN2.forward以注入语言标签（支持所有target_module）==========
    if per_sample_routing_enabled:
        logger.info("\n=== Enabling Per-Sample Group Routing ===")
        logger.info(f"  target_module: {args.target_module}")
        
        try:
            if hasattr(model, 'speech_encoder'):
                speech_encoder_routing = model.speech_encoder
            elif hasattr(model, 'model') and hasattr(model.model, 'speech_encoder'):
                speech_encoder_routing = model.model.speech_encoder
            else:
                raise AttributeError("Cannot find speech_encoder")
            
            # 根据target_module找到实际被替换的GroupSpecificFFN2模块
            target_gsffn2 = None
            
            if args.target_module == 'encoder_ffn2':
                # 原始行为：layers[11].ffn2
                if hasattr(speech_encoder_routing, 'encoder') and hasattr(speech_encoder_routing.encoder, 'layers'):
                    _enc_layers = speech_encoder_routing.encoder.layers
                elif hasattr(speech_encoder_routing, 'inner') and hasattr(speech_encoder_routing.inner, 'layers'):
                    _enc_layers = speech_encoder_routing.inner.layers
                elif hasattr(speech_encoder_routing, 'layers'):
                    _enc_layers = speech_encoder_routing.layers
                else:
                    raise AttributeError("Cannot find encoder layers")
                target_gsffn2 = _enc_layers[11].ffn2
                
            elif args.target_module == 'encoder_ffn1':
                # layers[11].ffn1
                if hasattr(speech_encoder_routing, 'encoder') and hasattr(speech_encoder_routing.encoder, 'layers'):
                    _enc_layers = speech_encoder_routing.encoder.layers
                elif hasattr(speech_encoder_routing, 'inner') and hasattr(speech_encoder_routing.inner, 'layers'):
                    _enc_layers = speech_encoder_routing.inner.layers
                elif hasattr(speech_encoder_routing, 'layers'):
                    _enc_layers = speech_encoder_routing.layers
                else:
                    raise AttributeError("Cannot find encoder layers")
                target_gsffn2 = _enc_layers[11].ffn1
                
            elif args.target_module == 'encoder_layer10_ffn2':
                # layers[10].ffn2
                if hasattr(speech_encoder_routing, 'encoder') and hasattr(speech_encoder_routing.encoder, 'layers'):
                    _enc_layers = speech_encoder_routing.encoder.layers
                elif hasattr(speech_encoder_routing, 'inner') and hasattr(speech_encoder_routing.inner, 'layers'):
                    _enc_layers = speech_encoder_routing.inner.layers
                elif hasattr(speech_encoder_routing, 'layers'):
                    _enc_layers = speech_encoder_routing.layers
                else:
                    raise AttributeError("Cannot find encoder layers")
                target_gsffn2 = _enc_layers[10].ffn2
                
            elif args.target_module == 'adapter_ffn':
                # speech_encoder.adapter.layers[0].ffn (HuggingFace) 或 adaptor_layers[0].ffn (fairseq2)
                if hasattr(speech_encoder_routing, 'adapter') and hasattr(speech_encoder_routing.adapter, 'layers'):
                    target_gsffn2 = speech_encoder_routing.adapter.layers[0].ffn
                elif hasattr(speech_encoder_routing, 'adaptor_layers') and len(speech_encoder_routing.adaptor_layers) > 0:
                    target_gsffn2 = speech_encoder_routing.adaptor_layers[0].ffn
                else:
                    raise AttributeError("Cannot find adapter module")
                    
            elif args.target_module == 'intermediate_ffn':
                target_gsffn2 = speech_encoder_routing.intermediate_ffn
            
            # 检查是否是GroupSpecificFFN2
            if target_gsffn2 is None or not hasattr(target_gsffn2, 'group1_ffn2'):
                logger.warning(f"Target module '{args.target_module}' is not GroupSpecificFFN2, per-sample routing disabled")
                logger.warning(f"  Module type: {type(target_gsffn2)}")
                per_sample_routing_enabled = False
            else:
                # 保存原始forward
                original_gsffn2_forward = target_gsffn2.forward
                
                # 创建wrapper以注入语言标签
                # 训练和eval都使用per-sample routing，各自维护独立计数器
                # 使用模运算自动处理epoch/run边界，无需手动重置
                _total_train_samples = len(sample_to_lang)
                _steps_per_epoch = max(1, _total_train_samples // args.batch_size)
                _total_eval_samples = len(eval_sample_to_lang)
                _eval_steps_per_run = max(1, _total_eval_samples // args.batch_size)

                class FFN2WithLangInjection:
                    def __init__(self, ffn2_module, sample_to_lang_map, eval_sample_to_lang_map,
                                 batch_size, steps_per_epoch, eval_steps_per_run):
                        self.ffn2 = ffn2_module
                        self.sample_to_lang = sample_to_lang_map
                        self.eval_sample_to_lang = eval_sample_to_lang_map
                        self.batch_size = batch_size
                        self.steps_per_epoch = steps_per_epoch
                        self.eval_steps_per_run = eval_steps_per_run
                        self.train_batch_idx = 0   # 只在train时自增
                        self.eval_batch_idx = 0    # 只在eval时自增，模运算自动回绕
                        self.debug_counter = 0
                        self.bem_total = 0
                        self.other_total = 0

                    def __call__(self, x, group=None):
                        batch_size_actual = x.shape[0]

                        # ─── Eval 模式：按语言id正确路由，eval counter独立计数 ───
                        if not self.ffn2.training:
                            if self.eval_sample_to_lang:
                                epoch_eval_batch_idx = self.eval_batch_idx % self.eval_steps_per_run
                                first_sample_idx = epoch_eval_batch_idx * self.batch_size
                                batch_langs = [
                                    self.eval_sample_to_lang.get(first_sample_idx + i, 'aeb')
                                    for i in range(batch_size_actual)
                                ]
                                self.eval_batch_idx += 1
                                return original_gsffn2_forward(x, group=None, langs=batch_langs)
                            else:
                                # 无eval manifest时才降级为shared分支（兜底）
                                return self.ffn2.shared_ffn2(x)

                        # ─── Train 模式：用模运算计算当前epoch内的batch位置 ───
                        epoch_batch_idx = self.train_batch_idx % self.steps_per_epoch
                        first_sample_idx = epoch_batch_idx * self.batch_size

                        batch_langs = []
                        for i in range(batch_size_actual):
                            lang = self.sample_to_lang.get(first_sample_idx + i, 'aeb')
                            batch_langs.append(lang)

                        self.train_batch_idx += 1
                        self.debug_counter += 1

                        # 路由统计日志（每3000步一次）
                        bem_count = sum(1 for lang in batch_langs if lang == 'bem')
                        other_count = len(batch_langs) - bem_count
                        self.bem_total += bem_count
                        self.other_total += other_count
                        if self.debug_counter % 3000 == 0:
                            current_epoch = self.train_batch_idx // self.steps_per_epoch
                            logger.info(
                                f"[Per-Sample Routing] train_step={self.train_batch_idx} "
                                f"epoch≈{current_epoch} epoch_batch={epoch_batch_idx}: "
                                f"this_batch langs={batch_langs} | "
                                f"cumulative bem={self.bem_total} other={self.other_total}"
                            )

                        return original_gsffn2_forward(x, group=None, langs=batch_langs)

                    def reset_batch_counter(self):
                        self.train_batch_idx = 0
                        self.debug_counter = 0
                        self.bem_total = 0
                        self.other_total = 0

                # 创建wrapper实例并替换forward方法
                ffn2_wrapper = FFN2WithLangInjection(
                    target_gsffn2,
                    sample_to_lang,
                    eval_sample_to_lang,
                    args.batch_size,
                    _steps_per_epoch,
                    _eval_steps_per_run,
                )
                target_gsffn2.forward = ffn2_wrapper
                
                logger.info(f"✓ Per-sample group routing enabled for target_module={args.target_module}")
                logger.info(f"  - Module type: {type(target_gsffn2).__name__}")
                logger.info(f"  - Batch size: {args.batch_size}")
                logger.info(f"  - Total samples: {len(sample_to_lang)}")
                logger.info("  - Each sample will use its own language-specific private branch")
                
        except Exception as e:
            logger.error(f"Failed to enable per-sample routing: {e}")
            import traceback
            traceback.print_exc()
            per_sample_routing_enabled = False
    
    # ========== 启用PCGrad梯度修正（修复集成问题）==========
    pcgrad_handler = None
    if args.use_pcgrad:
        logger.info("\n=== Enabling PCGrad Gradient Correction ===")
        logger.info("PCGrad will detect and resolve gradient conflicts between group1 and group2")
        
        pcgrad_handler = PCGradHandler(
            model=model,
            log_steps=args.pcgrad_log_steps,
            logger=logger
        )
        
        # 修复：必须在backward后、optimizer.step前调用PCGrad
        # 保存原始的_train_step
        original_train_step = finetune._train_step
        
        def train_step_with_pcgrad(batch):
            """
            修改后的train step，在正确位置应用PCGrad
            
            原始流程（在trainer内部）：
            1. model.zero_grad()
            2. forward() + loss计算
            3. loss.backward()  ← 梯度已计算
            4. optimizer.step()  ← 更新参数
            
            修改后（intercept）：
            1-3. 保持不变（在original_train_step内）
            3.5. PCGrad修正梯度  ← 在backward后、step前插入
            4. 保持不变
            
            实现方式：Hook到model的参数上，拦截step操作
            """
            # 方案：使用register_hook拦截参数
            # 但这里更简单的做法是：直接修改batch后调用PCGrad
            
            # 首先调用原始train_step（会执行forward, backward, step）
            result = original_train_step(batch)
            
            # ⚠️  问题：此时已经执行了step，PCGrad应该在step前应用！
            # 需要重新设计：不能使用这种方式
            
            return result
        
        # 更好的方案：Monkey-patch optimizer的step方法（整个过程中）
        # 但使用全局标志确保thread-safe
        original_optimizer_step = finetune.optimizer.step
        
        # 用来标记是否需要应用PCGrad的flag
        _pcgrad_should_apply = True
        
        def optimizer_step_with_pcgrad_guard(*args, **kwargs):
            """
            Guarded version of optimizer.step that applies PCGrad before stepping
            """
            nonlocal _pcgrad_should_apply
            
            if _pcgrad_should_apply:
                try:
                    # ✓ 此时梯度已从backward()计算好，即将执行step()
                    pcgrad_handler.apply_pcgrad()
                except Exception as e:
                    logger.warning(f"PCGrad application failed: {e}")
            
            # 调用原始step
            return original_optimizer_step(*args, **kwargs)
        
        # 替换optimizer.step（全局替换，所有train_step都会用到）
        finetune.optimizer.step = optimizer_step_with_pcgrad_guard
        
        logger.info("✓ PCGrad hook registered successfully")
        logger.info("  PCGrad will be applied before optimizer.step()")
        logger.info("  Gradient conflicts will be logged every {} steps".format(args.pcgrad_log_steps))
    
    # ========== per-sample routing 已通过模运算自动处理epoch边界，无需额外patch ==========
    # FFN2WithLangInjection 使用 train_batch_idx % steps_per_epoch 自动回绕
    # eval 模式下 ffn2.training=False，不自增计数器，不会污染训练计数
    
    # 设置分组学习率
    if args.two_stage_training:
        # 两阶段训练：只训练group参数，使用private_lr和private_weight_decay
        logger.info("\n=== Two-Stage Training: Configuring Private Parameters Only ===")
        
        private_lr = args.private_lr or 1e-4  # 默认1e-4
        private_wd = args.private_weight_decay  # 默认1e-3
        
        logger.info(f"Private LR: {private_lr}")
        logger.info(f"Private Weight Decay: {private_wd}")
        
        # 只配置group参数
        optimizer = finetune.optimizer
        if optimizer is not None:
            # 收集所有group参数
            group_params = []
            for param in model.parameters():
                if param.requires_grad and hasattr(param, 'param_group') and param.param_group == 'group':
                    group_params.append(param)
            
            # 获取原始优化器配置
            original_group = optimizer.param_groups[0]
            base_config = {k: v for k, v in original_group.items() if k not in ['params', 'lr', 'weight_decay']}
            
            # 重新构造param_groups：只有一个group（private）
            new_param_groups = [{
                'params': group_params,
                'lr': private_lr,
                'weight_decay': private_wd,
                'name': 'private_group_ffn2',
                **base_config
            }]
            
            optimizer.param_groups = new_param_groups
            logger.info(f"✓ Configured {len(group_params)} private parameters")
            logger.info(f"  - LR: {private_lr}")
            logger.info(f"  - Weight Decay: {private_wd}")
        else:
            logger.warning("Optimizer not yet created")
    
    elif args.load_checkpoint_path or args.use_grouped_lr:
        # 混合训练或分组学习率模式：同时训练shared和group，使用不同的学习率
        logger.info("\n=== Setting up Grouped Learning Rates ===")
        logger.info("Shared parameters and Group parameters will use different learning rates")
        
        group_lr = args.group_learning_rate or (args.learning_rate * 3)
        logger.info(f"Shared LR: {args.learning_rate}")
        logger.info(f"Group LR: {group_lr}")
        
        setup_grouped_learning_rate(
            model=model,
            finetune=finetune,
            shared_lr=args.learning_rate,
            group_lr=group_lr,
            logger=logger
        )
    
    # 开始训练
    logger.info("\n=== Starting Training ===")
    logger.info("Using official UnitYFinetune trainer")
    if per_sample_routing_enabled:
        logger.info("✓ Per-sample group routing enabled")
        logger.info("  Each sample will route to its own language-specific private branch")
    else:
        logger.info("⚠️  Using per-batch routing (fallback mode)")
    
    if args.use_pcgrad:
        logger.info("✓ PCGrad gradient conflict resolution enabled")
    
    logger.info("All batch structure and forward compatibility handled by trainer")
    
    finetune.run()
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"✓ Model saved to: {args.save_model_to}")
    
    if per_sample_routing_enabled:
        logger.info("\n=== Per-Sample Routing Statistics ===")
        logger.info(f"  Total samples processed: {len(sample_to_lang)}")
        logger.info(f"  Language distribution:")
        for lang, count in sorted(lang_counts.items()):
            percentage = 100.0 * count / len(sample_to_lang)
            logger.info(f"    {lang}: {count} ({percentage:.1f}%)")
    
    if pcgrad_handler is not None:
        logger.info("\n=== PCGrad Final Statistics ===")
        if pcgrad_handler.total_checks > 0:
            conflict_rate = 100.0 * pcgrad_handler.conflict_count / pcgrad_handler.total_checks
            avg_cosine = pcgrad_handler.cosine_sum / pcgrad_handler.total_checks
            logger.info(f"  Total gradient checks: {pcgrad_handler.total_checks}")
            logger.info(f"  Conflicts resolved: {pcgrad_handler.conflict_count} ({conflict_rate:.1f}%)")
            logger.info(f"  Average gradient cosine similarity: {avg_cosine:.4f}")


if __name__ == "__main__":
    main()
