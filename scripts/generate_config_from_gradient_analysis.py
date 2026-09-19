#!/usr/bin/env python3
"""
Generate training configuration based on gradient analysis results.

This script reads three gradient analysis JSON files and generates:
1. Language grouping from KMeans clustering
2. Shared parameter ratio based on gradient conflict analysis (delta)
3. Energy distribution ratio between language groups

Output: JSON configuration file for B1 model training

Usage:
python generate_config_from_gradient_analysis.py \
  --kmeans_result /path/to/method_a_cosine_grouping/results.json \
  --similarity_result /path/to/zero_shot/summary.json \
  --energy_result /path/to/method_b_subspace_similarity/results.json \
  --output_dir /path/to/config_result_based_on_gcc
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradientAnalysisConfigGenerator:
    """Generate B1 model training configuration from gradient analysis results."""
    
    def __init__(self, kmeans_path: Path, similarity_path: Path, energy_path: Path):
        """
        Initialize config generator with three analysis result files.
        
        Args:
            kmeans_path: Path to method_a_cosine_grouping/results.json
            similarity_path: Path to zero_shot/summary.json
            energy_path: Path to method_b_subspace_similarity/results.json
        """
        self.kmeans_path = Path(kmeans_path)
        self.similarity_path = Path(similarity_path)
        self.energy_path = Path(energy_path)
        
        self.kmeans_result = None
        self.similarity_result = None
        self.energy_result = None
        
        self._load_results()
    
    def _load_results(self):
        """Load all three result files."""
        logger.info("Loading gradient analysis results...")
        
        with open(self.kmeans_path, 'r', encoding='utf-8') as f:
            self.kmeans_result = json.load(f)
            logger.info(f"✓ Loaded KMeans clustering results from {self.kmeans_path}")
        
        with open(self.similarity_path, 'r', encoding='utf-8') as f:
            self.similarity_result = json.load(f)
            logger.info(f"✓ Loaded gradient similarity results from {self.similarity_path}")
        
        with open(self.energy_path, 'r', encoding='utf-8') as f:
            self.energy_result = json.load(f)
            logger.info(f"✓ Loaded energy distribution results from {self.energy_path}")
    
    def get_language_groups(self) -> Dict[int, List[str]]:
        """
        Extract language groups from KMeans clustering results.
        
        Returns:
            Dict mapping cluster_id -> list of language pairs
            Example: {0: ['bem_eng'], 1: ['aeb_eng', 'est_eng', 'gle_eng']}
        """
        logger.info("\n=== Step 1: Language Grouping ===")
        
        assignments = self.kmeans_result['kmeans_clustering']['assignments']
        groups = {}
        
        for lang, cluster_id in assignments.items():
            if cluster_id not in groups:
                groups[cluster_id] = []
            groups[cluster_id].append(lang)
        
        logger.info(f"Found {len(groups)} language groups:")
        for cluster_id, langs in sorted(groups.items()):
            logger.info(f"  Cluster {cluster_id}: {langs}")
        
        return groups
    
    def calculate_shared_ratio(self) -> Tuple[float, Dict]:
        """
        Calculate shared parameter ratio based on gradient similarity conflict.
        
        Formula:
            Sself = mean of diagonal elements (self-similarity)
            Scross = mean of off-diagonal elements (cross-language similarity)
            delta = Sself - Scross
        
        Thresholds:
            delta < 0.05: 75% shared, 25% group-specific
            0.05 <= delta < 0.15: 50% shared, 50% group-specific
            delta >= 0.15: 25% shared, 75% group-specific
        
        Returns:
            Tuple of (delta_value, config_dict)
        """
        logger.info("\n=== Step 2: Shared Parameter Ratio Calculation ===")
        
        # Extract diagonal elements (self-similarity)
        diagonal_values = []
        for lang_pair in self.similarity_result.keys():
            if lang_pair in self.similarity_result[lang_pair]:
                mean_val = self.similarity_result[lang_pair][lang_pair]['mean']
                diagonal_values.append(mean_val)
        
        sself = sum(diagonal_values) / len(diagonal_values) if diagonal_values else 0
        logger.info(f"Self-similarity (Sself) diagonal mean: {sself:.6f}")
        logger.info(f"  Diagonal values: {[f'{v:.6f}' for v in diagonal_values]}")
        
        # Extract off-diagonal elements (cross-language similarity)
        cross_values = []
        langs = sorted(self.similarity_result.keys())
        n_langs = len(langs)
        
        for i in range(n_langs):
            for j in range(i + 1, n_langs):
                lang_i = langs[i]
                lang_j = langs[j]
                
                # Cross-language similarity (could be from either direction)
                if lang_j in self.similarity_result[lang_i]:
                    mean_val = self.similarity_result[lang_i][lang_j]['mean']
                    cross_values.append(mean_val)
        
        scross = sum(cross_values) / len(cross_values) if cross_values else 0
        logger.info(f"Cross-similarity (Scross) mean: {scross:.6f}")
        logger.info(f"  Cross-language pairs: {len(cross_values)}")
        logger.info(f"  Cross-values: {[f'{v:.6f}' for v in cross_values]}")
        
        delta = sself - scross
        logger.info(f"\nConflict Score (delta = Sself - Scross): {delta:.6f}")
        
        # Determine shared ratio based on delta thresholds
        if delta < 0.05:
            shared_ratio = 0.75
            r_shared = 3072
            r_group = 512
            ratio_category = "75% Shared (Low Conflict)"
        elif delta < 0.15:
            shared_ratio = 0.50
            r_shared = 2048
            r_group = 1024
            ratio_category = "50% Shared (Medium Conflict)"
        else:
            shared_ratio = 0.25
            r_shared = 1024
            r_group = 1536
            ratio_category = "25% Shared (High Conflict)"
        
        logger.info(f"\nThreshold Decision: δ={delta:.6f} → {ratio_category}")
        logger.info(f"  Parameters: --r_shared {r_shared} --r_group {r_group}")
        
        config = {
            'shared_ratio': shared_ratio,
            'r_shared': r_shared,
            'r_group': r_group,
            'ratio_category': ratio_category,
            'delta': delta,
            'sself': sself,
            'scross': scross
        }
        
        return delta, config
    
    def calculate_energy_ratio(self, language_groups: Dict[int, List[str]]) -> Dict:
        """
        Calculate energy distribution ratio between language groups.
        
        Process:
            1. Sum top-k energy for each language
            2. Sum energies for each group
            3. Calculate group-level energy ratio
        
        Args:
            language_groups: Language grouping from get_language_groups()
        
        Returns:
            Dict with energy ratios and detailed breakdown
        """
        logger.info("\n=== Step 3: Energy Distribution Ratio ===")
        
        per_lang_energy = self.energy_result.get('per_language_energy_top_k', {})
        
        if not per_lang_energy:
            logger.warning("No per_language_energy_top_k found in results")
            return {}
        
        # Calculate energy sum for each language
        lang_energy_sums = {}
        logger.info("\nLanguage-level energy sums (top-k):")
        for lang, energy_list in per_lang_energy.items():
            energy_sum = sum(energy_list)
            lang_energy_sums[lang] = energy_sum
            logger.info(f"  {lang}: {energy_sum:.6f} (from {len(energy_list)} components)")
        
        # Calculate energy sum for each group
        group_energy_sums = {}
        logger.info("\nGroup-level energy sums:")
        for group_id, langs in language_groups.items():
            group_sum = sum(lang_energy_sums.get(lang, 0) for lang in langs)
            group_energy_sums[group_id] = group_sum
            logger.info(f"  Cluster {group_id}: {group_sum:.6f} (languages: {langs})")
        
        # Calculate total energy and ratios
        total_energy = sum(group_energy_sums.values())
        logger.info(f"\nTotal energy: {total_energy:.6f}")
        
        group_ratios = {}
        logger.info("\nEnergy ratios per group:")
        for group_id, energy_sum in group_energy_sums.items():
            ratio = energy_sum / total_energy if total_energy > 0 else 0
            group_ratios[group_id] = ratio
            langs = language_groups[group_id]
            logger.info(f"  Cluster {group_id} ({', '.join(langs)}): {ratio:.6f} ({ratio*100:.1f}%)")
        
        # Sort groups by ID for consistent ordering
        sorted_group_ids = sorted(group_ratios.keys())
        
        config = {
            'language_groups': language_groups,
            'lang_energy_sums': lang_energy_sums,
            'group_energy_sums': group_energy_sums,
            'total_energy': total_energy,
            'group_ratios': {str(gid): group_ratios[gid] for gid in sorted_group_ids},
            'group_ratios_list': [group_ratios[gid] for gid in sorted_group_ids]
        }
        
        return config
    
    def generate_config(self) -> Dict:
        """
        Generate complete training configuration.
        
        Returns:
            Dict with all configuration parameters
        """
        logger.info("\n" + "="*80)
        logger.info("GRADIENT ANALYSIS CONFIGURATION GENERATION")
        logger.info("="*80)
        
        # Step 1: Language grouping
        language_groups = self.get_language_groups()
        
        # Step 2: Shared ratio
        delta, shared_config = self.calculate_shared_ratio()
        
        # Step 3: Energy ratio
        energy_config = self.calculate_energy_ratio(language_groups)
        
        # Combine all configurations
        full_config = {
            'metadata': {
                'kmeans_file': str(self.kmeans_path),
                'similarity_file': str(self.similarity_path),
                'energy_file': str(self.energy_path)
            },
            'language_grouping': {
                'groups': language_groups,
                'summary': self._summarize_groups(language_groups)
            },
            'shared_parameter_config': shared_config,
            'energy_distribution_config': energy_config,
            'training_parameters': self._generate_training_params(shared_config, energy_config)
        }
        
        return full_config
    
    def _summarize_groups(self, language_groups: Dict[int, List[str]]) -> str:
        """Generate a human-readable summary of language groups."""
        summaries = []
        for group_id, langs in sorted(language_groups.items()):
            summaries.append(f"Group {group_id}: {', '.join(langs)}")
        return " | ".join(summaries)
    
    def _generate_training_params(self, shared_config: Dict, energy_config: Dict) -> Dict:
        """
        Generate recommended training parameters based on configs.
        
        Returns:
            Dict of training parameters ready for command line
        """
        params = {
            'r_shared': shared_config['r_shared'],
            'r_group': shared_config['r_group'],
            'shared_ratio': shared_config['shared_ratio'],
            'group_energy_ratios': energy_config.get('group_ratios_list', [])
        }
        
        return params
    
    def save_config(self, config: Dict, output_dir: Path) -> Path:
        """
        Save configuration to output directory.
        
        Args:
            config: Configuration dict
            output_dir: Output directory path
        
        Returns:
            Path to saved config file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = output_dir / 'config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Configuration saved to {config_file}")
        
        # Also save a human-readable summary
        summary_file = output_dir / 'config_summary.txt'
        self._save_summary(config, summary_file)
        
        return config_file
    
    def _save_summary(self, config: Dict, summary_file: Path):
        """Save human-readable summary."""
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING CONFIGURATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Language Grouping
            f.write("1. LANGUAGE GROUPING\n")
            f.write("-" * 80 + "\n")
            f.write(f"Summary: {config['language_grouping']['summary']}\n\n")
            
            for group_id, langs in config['language_grouping']['groups'].items():
                f.write(f"  Group {group_id}: {', '.join(langs)}\n")
            
            # Shared Parameter Config
            f.write("\n\n2. SHARED PARAMETER CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            shared_cfg = config['shared_parameter_config']
            f.write(f"Conflict Score (δ): {shared_cfg['delta']:.6f}\n")
            f.write(f"Self-similarity (Sself): {shared_cfg['sself']:.6f}\n")
            f.write(f"Cross-similarity (Scross): {shared_cfg['scross']:.6f}\n")
            f.write(f"Decision: {shared_cfg['ratio_category']}\n")
            f.write(f"Shared Ratio: {shared_cfg['shared_ratio']*100:.0f}%\n")
            f.write(f"Training Parameters: --r_shared {shared_cfg['r_shared']} --r_group {shared_cfg['r_group']}\n")
            
            # Energy Distribution
            f.write("\n\n3. ENERGY DISTRIBUTION CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            energy_cfg = config['energy_distribution_config']
            
            f.write("\nLanguage-level energy:\n")
            for lang, energy in energy_cfg['lang_energy_sums'].items():
                f.write(f"  {lang}: {energy:.6f}\n")
            
            f.write("\nGroup-level energy:\n")
            for group_id, langs in config['language_grouping']['groups'].items():
                if str(group_id) in energy_cfg['group_ratios']:
                    ratio = energy_cfg['group_ratios'][str(group_id)]
                    f.write(f"  Group {group_id} ({', '.join(langs)}): {ratio:.6f} ({ratio*100:.1f}%)\n")
            
            # Training Parameters
            f.write("\n\n4. RECOMMENDED TRAINING PARAMETERS\n")
            f.write("-" * 80 + "\n")
            train_params = config['training_parameters']
            f.write(f"--r_shared {train_params['r_shared']}\n")
            f.write(f"--r_group {train_params['r_group']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"✓ Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate B1 model training configuration from gradient analysis results"
    )
    
    parser.add_argument(
        '--kmeans_result',
        type=Path,
        default=Path('/224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/zero_shot_ad/method_a_cosine_grouping/results.json'),
        help='Path to KMeans clustering results (method_a_cosine_grouping/results.json)'
    )
    
    parser.add_argument(
        '--similarity_result',
        type=Path,
        default=Path('/224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/zero_shot/summary.json'),
        help='Path to gradient similarity results (zero_shot/summary.json)'
    )
    
    parser.add_argument(
        '--energy_result',
        type=Path,
        default=Path('/224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/zero_shot_ad/method_b_subspace_similarity/results.json'),
        help='Path to energy distribution results (method_b_subspace_similarity/results.json)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('/224040284/workspace/seamless_communication/src/seamless_communication/output/grad_analysis/config_result_based_on_gcc'),
        help='Output directory for configuration files'
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    for path, name in [
        (args.kmeans_result, 'KMeans result'),
        (args.similarity_result, 'Similarity result'),
        (args.energy_result, 'Energy result')
    ]:
        if not path.exists():
            logger.error(f"✗ {name} not found: {path}")
            return 1
    
    # Generate configuration
    generator = GradientAnalysisConfigGenerator(
        args.kmeans_result,
        args.similarity_result,
        args.energy_result
    )
    
    config = generator.generate_config()
    config_file = generator.save_config(config, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("CONFIGURATION GENERATION COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
