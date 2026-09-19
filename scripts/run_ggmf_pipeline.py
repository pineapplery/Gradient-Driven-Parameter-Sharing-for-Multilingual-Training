#!/usr/bin/env python3
"""
GGMF (Gradient-Guided Multilingual Finetuning) 一体化流程脚本

完整流程：
1. 训练基础统一模型（Unified Model）
2. 梯度冲突分析
3. 高级梯度分析
4. 生成配置文件
5. 训练GGMF模型（B1架构）
6. 推理生成翻译结果
7. 评估翻译质量

特性：
- 自动创建目录结构
- 断点续训（如果checkpoint存在则跳过）
- 并行推理（多语言同时推理）
- 日志记录
- 错误处理

Usage:
python run_ggmf_pipeline.py \
  --root_dir /path/to/GGMF_result \
  --lang_pairs aeb_eng,bem_eng,est_eng,gle_eng \
  --data_dir /path/to/manifests \
  --train_manifest /path/to/train_all_manifest.json \
  --eval_manifest /path/to/valid_all_manifest.json \
  --test_file_pattern /path/to/test_{lang_pair}_npy.tsv
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GGMFPipeline:
    """GGMF一体化流程管理器"""
    
    def __init__(self, args):
        self.args = args
        self.root_dir = Path(args.root_dir)
        self.lang_pairs = args.lang_pairs.split(',')
        
        # 创建目录结构
        self.setup_directories()
        
        # 设置日志文件
        self.setup_logging()
        
        # 记录配置信息
        self.log_config()
    
    def setup_directories(self):
        """创建所需的目录结构"""
        logger.info("=== Setting up directory structure ===")
        
        self.dirs = {
            'root': self.root_dir,
            'checkpoint': self.root_dir / 'checkpoint',
            'checkpoint_unify': self.root_dir / 'checkpoint' / 'unify',
            'grad_analysis': self.root_dir / 'grad_analysis',
            'grad_analysis_zero_shot': self.root_dir / 'grad_analysis' / 'zero_shot',
            'grad_analysis_zero_shot_ad': self.root_dir / 'grad_analysis' / 'zero_shot_ad',
            'grad_analysis_config': self.root_dir / 'grad_analysis' / 'config_result_based_on_gcc',
            'group_finetuned': self.root_dir / 'group_finetuned',
            'logs': self.root_dir / 'logs',
        }
        
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created/verified directory: {path}")
        
        logger.info(f"Root directory: {self.root_dir}")
    
    def setup_logging(self):
        """设置日志文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.dirs['logs'] / f'ggmf_pipeline_{timestamp}.log'
        
        # 添加文件handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_file}")
    
    def log_config(self):
        """记录配置信息"""
        logger.info("\n" + "="*80)
        logger.info("GGMF Pipeline Configuration")
        logger.info("="*80)
        logger.info(f"Root directory: {self.root_dir}")
        logger.info(f"Language pairs: {self.lang_pairs}")
        logger.info(f"Data directory: {self.args.data_dir}")
        logger.info(f"Train manifest: {self.args.train_manifest}")
        logger.info(f"Eval manifest: {self.args.eval_manifest}")
        logger.info(f"Test file pattern: {self.args.test_file_pattern}")
        logger.info(f"Skip existing: {self.args.skip_existing}")
        logger.info(f"Parallel inference: {self.args.parallel_inference}")
        logger.info("="*80 + "\n")
    
    def run_command(self, cmd: List[str], step_name: str, log_file: Path = None, env: Dict = None) -> bool:
        """
        执行命令并记录输出
        
        Args:
            cmd: 命令列表
            step_name: 步骤名称
            log_file: 日志文件路径
            env: 环境变量
        
        Returns:
            bool: 是否成功
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        if log_file is None:
            log_file = self.dirs['logs'] / f"{step_name.replace(' ', '_').lower()}.log"
        
        try:
            # 合并环境变量
            run_env = os.environ.copy()
            if env:
                run_env.update(env)
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=run_env,
                    bufsize=1
                )
                
                # 实时输出并写入日志
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                process.wait()
            
            if process.returncode == 0:
                logger.info(f"✓ {step_name} completed successfully")
                return True
            else:
                logger.error(f"✗ {step_name} failed with return code {process.returncode}")
                logger.error(f"Check log file: {log_file}")
                return False
        
        except Exception as e:
            logger.error(f"✗ {step_name} failed with exception: {e}")
            return False
    
    def step1_train_unified_model(self) -> bool:
        """步骤1：训练基础统一模型"""
        checkpoint_path = self.dirs['checkpoint_unify'] / 'unify_cp.pt'
        
        if self.args.skip_existing and checkpoint_path.exists():
            logger.info(f"✓ Unified checkpoint already exists: {checkpoint_path}")
            logger.info("  Skipping training (use --no-skip-existing to force retrain)")
            return True
        
        cmd = [
            'python', '/224040284/workspace/seamless_communication/src/seamless_communication/scripts/finetune_multilang.py',
            '--train_dataset', self.args.train_manifest,
            '--eval_dataset', self.args.eval_manifest,
            '--model_name', 'seamlessM4T_medium',
            '--save_model_to', str(checkpoint_path),
            '--batch_size', '8',
            '--max_epochs', '30',
            '--learning_rate', '5e-5',
            '--use_fbank',
            '--freeze_encoder_except_last_n', '2',
            '--device', 'cuda',
            '--patience', '10',
            '--eval_steps', '5000',
            '--warmup_steps', '1000',
            '--log_steps', '5000',
        ]
        
        return self.run_command(cmd, "Step 1: Train Unified Model")
    
    def step2_gradient_conflict_analysis(self) -> bool:
        """步骤2：梯度冲突分析"""
        checkpoint_path = self.dirs['checkpoint_unify'] / 'unify_cp.pt'
        
        if not checkpoint_path.exists():
            logger.error(f"✗ Unified checkpoint not found: {checkpoint_path}")
            return False
        
        # 检查是否已完成
        summary_file = self.dirs['grad_analysis_zero_shot'] / 'summary.json'
        if self.args.skip_existing and summary_file.exists():
            logger.info(f"✓ Gradient conflict analysis already completed")
            logger.info(f"  Output: {summary_file}")
            return True
        
        env = {
            'CUDA_VISIBLE_DEVICES': '1',
            'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '8',
        }
        
        cmd = [
            'python', '/224040284/workspace/seamless_communication/src/seamless_communication/scripts/gradient_conflict_analysis.py',
            '--manifest_dir', self.args.data_dir,
            '--lang_pairs', self.args.lang_pairs,
            '--split', 'train',
            '--checkpoint', str(checkpoint_path),
            '--output_dir', str(self.dirs['grad_analysis_zero_shot']),
            '--batch_size', '4',
            '--update_freq', '4',
            '--max_samples', '7000',
            '--proj_dim', '256',
            '--proj_only_last_n_layers', '2',
            '--param_level_norm',
            '--layer_level_norm',
            '--per_layer_analysis',
            '--freeze_encoder_except_last_n', '2',
            '--num_workers', '8',
        ]
        
        return self.run_command(cmd, "Step 2: Gradient Conflict Analysis", env=env)
    
    def step3_advanced_gradient_analysis(self) -> bool:
        """步骤3：高级梯度分析"""
        checkpoint_path = self.dirs['checkpoint_unify'] / 'unify_cp.pt'
        
        if not checkpoint_path.exists():
            logger.error(f"✗ Unified checkpoint not found: {checkpoint_path}")
            return False
        
        # 检查是否已完成
        method_a_result = self.dirs['grad_analysis_zero_shot_ad'] / 'method_a_cosine_grouping' / 'results.json'
        method_b_result = self.dirs['grad_analysis_zero_shot_ad'] / 'method_b_subspace_similarity' / 'results.json'
        
        if self.args.skip_existing and method_a_result.exists() and method_b_result.exists():
            logger.info(f"✓ Advanced gradient analysis already completed")
            return True
        
        env = {
            'CUDA_VISIBLE_DEVICES': '1',
            'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '8',
        }
        
        cmd = [
            'python', '/224040284/workspace/seamless_communication/src/seamless_communication/scripts/gradient_analysis_advanced.py',
            '--manifest_dir', self.args.data_dir,
            '--lang_pairs', self.args.lang_pairs,
            '--split', 'train',
            '--checkpoint', str(checkpoint_path),
            '--output_dir', str(self.dirs['grad_analysis_zero_shot_ad']),
            '--batch_size', '4',
            '--update_freq', '4',
            '--max_samples', '7000',
            '--proj_dim', '256',
            '--proj_only_last_n_layers', '2',
            '--param_level_norm',
            '--layer_level_norm',
            '--freeze_encoder_except_last_n', '2',
            '--num_workers', '8',
        ]
        
        return self.run_command(cmd, "Step 3: Advanced Gradient Analysis", env=env)
    
    def step4_generate_config(self) -> bool:
        """步骤4：生成配置文件"""
        kmeans_result = self.dirs['grad_analysis_zero_shot_ad'] / 'method_a_cosine_grouping' / 'results.json'
        similarity_result = self.dirs['grad_analysis_zero_shot'] / 'summary.json'
        energy_result = self.dirs['grad_analysis_zero_shot_ad'] / 'method_b_subspace_similarity' / 'results.json'
        
        # 验证输入文件存在
        for file_path in [kmeans_result, similarity_result, energy_result]:
            if not file_path.exists():
                logger.error(f"✗ Required file not found: {file_path}")
                return False
        
        # 检查是否已完成
        config_file = self.dirs['grad_analysis_config'] / 'config.json'
        if self.args.skip_existing and config_file.exists():
            logger.info(f"✓ Config already generated: {config_file}")
            return True
        
        cmd = [
            'python', '/224040284/generate_config_from_gradient_analysis.py',
            '--kmeans_result', str(kmeans_result),
            '--similarity_result', str(similarity_result),
            '--energy_result', str(energy_result),
            '--output_dir', str(self.dirs['grad_analysis_config']),
        ]
        
        return self.run_command(cmd, "Step 4: Generate Config")
    
    def extract_config_params(self) -> Tuple[str, str, str, str]:
        """
        从config.json提取训练参数
        
        Returns:
            (group_assignments, group_energy_ratios, r_shared, r_group)
        """
        config_file = self.dirs['grad_analysis_config'] / 'config.json'
        
        if not config_file.exists():
            logger.error(f"✗ Config file not found: {config_file}")
            return None, None, None, None
        
        logger.info(f"Reading config from: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 提取语言分组
        groups = config['language_grouping']['groups']
        group_assignments = []
        
        for group_id, langs in groups.items():
            # 确定组标签：cluster 0 -> g2, cluster 1 -> g1
            group_label = 'g2' if group_id == '0' else 'g1'
            for lang in langs:
                # 移除_eng后缀
                lang_code = lang.replace('_eng', '')
                group_assignments.append(f"{lang_code}:{group_label}")
        
        group_assignments_str = ','.join(group_assignments)
        
        # 提取能量比例（注意映射：cluster 0 -> g2, cluster 1 -> g1）
        energy_ratios = config['energy_distribution_config']['group_ratios']
        g1_ratio = float(energy_ratios.get('1', 0.723))
        g2_ratio = float(energy_ratios.get('0', 0.277))
        group_energy_ratios_str = f"g1:{g1_ratio:.6f},g2:{g2_ratio:.6f}"
        
        # 提取共享参数比例
        r_shared = str(config['shared_parameter_config']['r_shared'])
        r_group = str(config['shared_parameter_config']['r_group'])
        
        logger.info(f"Extracted config:")
        logger.info(f"  group_assignments: {group_assignments_str}")
        logger.info(f"  group_energy_ratios: {group_energy_ratios_str}")
        logger.info(f"  r_shared: {r_shared}")
        logger.info(f"  r_group: {r_group}")
        
        return group_assignments_str, group_energy_ratios_str, r_shared, r_group
    
    def step5_train_ggmf_model(self) -> bool:
        """步骤5：训练GGMF模型（B1架构）"""
        ggmf_checkpoint = self.dirs['checkpoint'] / 'GGMF_checkpoint.pt'
        unify_checkpoint = self.dirs['checkpoint_unify'] / 'unify_cp.pt'
        
        if not unify_checkpoint.exists():
            logger.error(f"✗ Unified checkpoint not found: {unify_checkpoint}")
            return False
        
        # 检查是否已完成
        if self.args.skip_existing and ggmf_checkpoint.exists():
            logger.info(f"✓ GGMF checkpoint already exists: {ggmf_checkpoint}")
            return True
        
        # 提取配置参数
        group_assignments, group_energy_ratios, r_shared, r_group = self.extract_config_params()
        
        if group_assignments is None:
            logger.error("✗ Failed to extract config parameters")
            return False
        
        cmd = [
            'python', '/224040284/workspace/seamless_communication/src/seamless_communication/scripts/finetune_B1_trainer.py',
            '--train_dataset', self.args.train_manifest,
            '--eval_dataset', self.args.eval_manifest,
            '--model_name', 'seamlessM4T_medium',
            '--save_model_to', str(ggmf_checkpoint),
            '--batch_size', '4',
            '--max_epochs', '20',
            '--device', 'cuda',
            '--seed', '2343',
            '--eval_steps', '5000',
            '--use_fbank',
            '--log_steps', '5000',
            '--warmup_steps', '2000',
            '--patience', '15',
            '--load_checkpoint_path', str(unify_checkpoint),
            '--learning_rate', '4e-5',
            '--group_learning_rate', '1e-4',
            '--max_src_tokens', '15000',
            '--private_weight_decay', '0.05',
            '--dropout_rate', '0.05',
            '--r_shared', r_shared,
            '--r_group', r_group,
            '--group_assignments', group_assignments,
            '--group_energy_ratios', group_energy_ratios,
            '--init_strategy', 'residual',
        ]
        
        return self.run_command(cmd, "Step 5: Train GGMF Model")
    
    def step6_inference_single_language(self, lang_pair: str) -> bool:
        """
        为单个语言对执行推理
        
        Args:
            lang_pair: 语言对，如 'aeb_eng'
        
        Returns:
            bool: 是否成功
        """
        ggmf_checkpoint = self.dirs['checkpoint'] / 'GGMF_checkpoint.pt'
        
        if not ggmf_checkpoint.exists():
            logger.error(f"✗ GGMF checkpoint not found: {ggmf_checkpoint}")
            return False
        
        # 解析语言对
        src_lang, tgt_lang = lang_pair.split('_')
        
        # 生成测试文件路径
        test_file = self.args.test_file_pattern.format(lang_pair=lang_pair)
        output_file = self.dirs['group_finetuned'] / f'output_{lang_pair}_npy_results.tsv'
        
        # 检查是否已完成
        if self.args.skip_existing and output_file.exists():
            logger.info(f"✓ Inference for {lang_pair} already completed: {output_file}")
            return True
        
        # 验证测试文件存在
        if not Path(test_file).exists():
            logger.error(f"✗ Test file not found: {test_file}")
            return False
        
        env = {
            'PYTHONPATH': '/224040284/workspace/seamless_communication/src/seamless_communication/models/seamless_m4t_medium:' + os.environ.get('PYTHONPATH', ''),
        }
        
        cmd = [
            'python', '/224040284/workspace/seamless_communication/src/seamless_communication/cli/m4t/predict/predict_b1_translator.py',
            test_file,
            '--task', 'S2TT',
            '--tgt_lang', tgt_lang,
            '--src_lang', src_lang,
            '--load_checkpoint', str(ggmf_checkpoint),
            '--output_path', str(output_file),
        ]
        
        log_file = self.dirs['logs'] / f'inference_{lang_pair}.log'
        return self.run_command(cmd, f"Step 6: Inference ({lang_pair})", log_file=log_file, env=env)
    
    def step6_inference_all_languages(self) -> bool:
        """步骤6：为所有语言执行推理"""
        logger.info(f"\n{'='*80}")
        logger.info("Step 6: Inference for all languages")
        logger.info(f"{'='*80}")
        
        if self.args.parallel_inference:
            logger.info(f"Running inference in parallel for {len(self.lang_pairs)} languages")
            
            # 并行执行
            with mp.Pool(processes=min(len(self.lang_pairs), 4)) as pool:
                results = pool.map(self.step6_inference_single_language, self.lang_pairs)
            
            if all(results):
                logger.info("✓ All inference tasks completed successfully")
                return True
            else:
                logger.error("✗ Some inference tasks failed")
                for i, (lang_pair, success) in enumerate(zip(self.lang_pairs, results)):
                    if not success:
                        logger.error(f"  Failed: {lang_pair}")
                return False
        else:
            logger.info(f"Running inference sequentially for {len(self.lang_pairs)} languages")
            
            # 串行执行
            for lang_pair in self.lang_pairs:
                if not self.step6_inference_single_language(lang_pair):
                    logger.error(f"✗ Inference failed for {lang_pair}")
                    return False
            
            logger.info("✓ All inference tasks completed successfully")
            return True
    
    def step7_evaluation(self) -> bool:
        """步骤7：评估翻译质量"""
        # 验证推理结果存在
        missing_files = []
        for lang_pair in self.lang_pairs:
            output_file = self.dirs['group_finetuned'] / f'output_{lang_pair}_npy_results.tsv'
            if not output_file.exists():
                missing_files.append(str(output_file))
        
        if missing_files:
            logger.error(f"✗ Inference output files missing:")
            for f in missing_files:
                logger.error(f"  {f}")
            return False
        
        # 检查是否已完成
        eval_result = self.dirs['group_finetuned'] / 'evaluation_results.json'
        if self.args.skip_existing and eval_result.exists():
            logger.info(f"✓ Evaluation already completed: {eval_result}")
            return True
        
        cmd = [
            'python', '/224040284/evaluate_multilang_s2t.py',
            '--pred_dir', str(self.dirs['group_finetuned']),
            '--output_dir', str(self.dirs['group_finetuned']),
        ]
        
        return self.run_command(cmd, "Step 7: Evaluation")
    
    def run_pipeline(self):
        """运行完整流程"""
        logger.info("\n" + "="*80)
        logger.info("STARTING GGMF PIPELINE")
        logger.info("="*80 + "\n")
        
        start_time = datetime.now()
        
        steps = [
            ("Step 1: Train Unified Model", self.step1_train_unified_model),
            ("Step 2: Gradient Conflict Analysis", self.step2_gradient_conflict_analysis),
            ("Step 3: Advanced Gradient Analysis", self.step3_advanced_gradient_analysis),
            ("Step 4: Generate Config", self.step4_generate_config),
            ("Step 5: Train GGMF Model", self.step5_train_ggmf_model),
            ("Step 6: Inference", self.step6_inference_all_languages),
            ("Step 7: Evaluation", self.step7_evaluation),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*80}")
            logger.info(f"EXECUTING: {step_name}")
            logger.info(f"{'='*80}")
            
            try:
                success = step_func()
                
                if not success:
                    logger.error(f"\n{'='*80}")
                    logger.error(f"PIPELINE FAILED AT: {step_name}")
                    logger.error(f"{'='*80}")
                    logger.error(f"Please check the logs in: {self.dirs['logs']}")
                    return False
                
            except Exception as e:
                logger.error(f"\n{'='*80}")
                logger.error(f"PIPELINE ERROR AT: {step_name}")
                logger.error(f"{'='*80}")
                logger.error(f"Exception: {e}", exc_info=True)
                return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total duration: {duration}")
        logger.info(f"Results saved to: {self.root_dir}")
        logger.info(f"Logs saved to: {self.dirs['logs']}")
        logger.info("="*80 + "\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="GGMF (Gradient-Guided Multilingual Finetuning) 一体化流程脚本"
    )
    
    # 必需参数
    parser.add_argument(
        '--root_dir',
        type=str,
        default='/224040284/workspace/seamless_communication/src/seamless_communication/output/GGMF_result',
        help='根目录路径（默认：/224040284/workspace/seamless_communication/src/seamless_communication/output/GGMF_result）'
    )
    
    parser.add_argument(
        '--lang_pairs',
        type=str,
        default='aeb_eng,bem_eng,est_eng,gle_eng',
        help='语言对列表，逗号分隔（默认：aeb_eng,bem_eng,est_eng,gle_eng）'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests',
        help='数据集目录（默认：/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests）'
    )
    
    parser.add_argument(
        '--train_manifest',
        type=str,
        default='/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests/rebalance_manifest/train_all_manifest.json',
        help='训练manifest文件'
    )
    
    parser.add_argument(
        '--eval_manifest',
        type=str,
        default='/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/manifests/rebalance_manifest/valid_all_manifest.json',
        help='验证manifest文件'
    )
    
    parser.add_argument(
        '--test_file_pattern',
        type=str,
        default='/224040284/code/fairseq-main/examples/speech_text_joint_to_text/data_big/test_{lang_pair}_npy.tsv',
        help='测试文件路径模板，使用{lang_pair}作为占位符'
    )
    
    # 可选参数
    parser.add_argument(
        '--skip-existing',
        dest='skip_existing',
        action='store_true',
        default=True,
        help='跳过已存在的checkpoint和结果（默认：True）'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        dest='skip_existing',
        action='store_false',
        help='不跳过已存在的checkpoint，强制重新训练'
    )
    
    parser.add_argument(
        '--parallel-inference',
        dest='parallel_inference',
        action='store_true',
        default=True,
        help='并行执行推理（默认：True）'
    )
    
    parser.add_argument(
        '--no-parallel-inference',
        dest='parallel_inference',
        action='store_false',
        help='串行执行推理'
    )
    
    args = parser.parse_args()
    
    # 创建pipeline实例并运行
    pipeline = GGMFPipeline(args)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
