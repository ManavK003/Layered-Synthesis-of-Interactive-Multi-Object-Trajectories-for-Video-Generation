"""
Complete Multi-Object Video Generation Evaluation Pipeline
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from multi_object_metrics import MultiObjectMetrics, VideoQualityMetrics
from animal_motion_prior import AnimalMotionPrior


class MultiObjectEvaluator:
    """
    Comprehensive evaluation pipeline for multi-object video generation
    """
    
    def __init__(
        self,
        test_dataset_path: str,
        motion_priors_path: str = "./data/animal_motion_cache.pkl",
        results_dir: str = "./results"
    ):
        self.test_dataset_path = test_dataset_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics_calculator = MultiObjectMetrics()
        self.video_metrics = VideoQualityMetrics()
        
        # Load motion priors
        self.motion_prior = AnimalMotionPrior()
        self.motion_prior.load_cache()
        
        # Load test dataset
        self.test_data = self._load_test_dataset()
        
        print(f"Loaded {len(self.test_data)} test samples")
    
    def _load_test_dataset(self) -> List[Dict]:
        """
        Load test dataset (Animal Kingdom or custom)
        
        Returns:
            test_data: List of test samples
        """
        test_file = os.path.join(self.test_dataset_path, "test_annotations.json")
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        else:
            print("Warning: Test annotations not found, creating dummy data")
            test_data = self._create_dummy_test_data()
        
        return test_data
    
    def _create_dummy_test_data(self) -> List[Dict]:
        """Create dummy test data for testing"""
        dummy_data = []
        
        scenarios = [
            {
                'prompt': 'A lion chasing a zebra in the savanna',
                'objects': [
                    {'name': 'lion', 'species': 'lion', 'priority': 0},
                    {'name': 'zebra', 'species': 'zebra', 'priority': 1}
                ]
            },
            {
                'prompt': 'Two elephants walking side by side',
                'objects': [
                    {'name': 'elephant1', 'species': 'elephant', 'priority': 0},
                    {'name': 'elephant2', 'species': 'elephant', 'priority': 0}
                ]
            },
            {
                'prompt': 'A cheetah hunting in the grassland',
                'objects': [
                    {'name': 'cheetah', 'species': 'cheetah', 'priority': 0}
                ]
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            # Generate random ground truth trajectories
            gt_trajectories = []
            for obj in scenario['objects']:
                num_frames = 24
                traj = []
                for t in range(num_frames):
                    alpha = t / num_frames
                    x1 = 0.1 + alpha * 0.6 + np.random.randn() * 0.02
                    y1 = 0.2 + alpha * 0.4 + np.random.randn() * 0.02
                    w = 0.15 + np.random.randn() * 0.01
                    h = 0.15 + np.random.randn() * 0.01
                    
                    bbox = [x1, y1, x1 + w, y1 + h]
                    traj.append(bbox)
                
                gt_trajectories.append(traj)
            
            dummy_data.append({
                'id': f'test_{i:03d}',
                'prompt': scenario['prompt'],
                'objects': scenario['objects'],
                'gt_trajectories': gt_trajectories
            })
        
        return dummy_data
    
    def evaluate_single_sample(
        self,
        pred_objects: List[List[torch.Tensor]],
        gt_objects: List[List[torch.Tensor]],
        species_info: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a single generated video
        
        Args:
            pred_objects: Predicted trajectories
            gt_objects: Ground truth trajectories
            species_info: Species labels
            
        Returns:
            metrics: Dictionary of metric values
        """
        metrics = self.metrics_calculator.compute_all_metrics(
            pred_objects=pred_objects,
            gt_objects=gt_objects,
            species_info=species_info,
            motion_priors=self.motion_prior.motion_statistics
        )
        
        return metrics
    
    def evaluate_method(
        self,
        method_name: str,
        generated_videos_dir: str
    ) -> Dict[str, any]:
        """
        Evaluate a complete method on test set
        
        Args:
            method_name: Name of the method being evaluated
            generated_videos_dir: Directory containing generated videos
            
        Returns:
            results: Aggregated results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}\n")
        
        all_metrics = []
        per_category_metrics = {
            'single_object': [],
            'two_objects': [],
            'multi_objects': []
        }
        
        for sample_idx, sample in enumerate(tqdm(self.test_data, desc="Evaluating")):
            sample_id = sample['id']
            
            # Load predicted trajectories
            pred_file = os.path.join(
                generated_videos_dir, 
                f"{sample_id}_predictions.json"
            )
            
            if not os.path.exists(pred_file):
                print(f"Warning: Predictions not found for {sample_id}")
                continue
            
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            # Convert to tensors
            pred_objects = [
                [torch.tensor(bbox) for bbox in traj]
                for traj in predictions['trajectories']
            ]
            
            gt_objects = [
                [torch.tensor(bbox) for bbox in traj]
                for traj in sample['gt_trajectories']
            ]
            
            species_info = [obj['species'] for obj in sample['objects']]
            
            # Compute metrics
            metrics = self.evaluate_single_sample(
                pred_objects, 
                gt_objects, 
                species_info
            )
            
            all_metrics.append(metrics)
            
            # Categorize by number of objects
            num_objects = len(pred_objects)
            if num_objects == 1:
                per_category_metrics['single_object'].append(metrics)
            elif num_objects == 2:
                per_category_metrics['two_objects'].append(metrics)
            else:
                per_category_metrics['multi_objects'].append(metrics)
        
        # Aggregate results
        results = self._aggregate_metrics(
            all_metrics, 
            per_category_metrics,
            method_name
        )
        
        # Save results
        self._save_results(results, method_name)
        
        # Generate visualizations
        self._generate_visualizations(results, method_name)
        
        return results
    
    def _aggregate_metrics(
        self,
        all_metrics: List[Dict],
        per_category_metrics: Dict[str, List[Dict]],
        method_name: str
    ) -> Dict:
        """Aggregate metrics across all samples"""
        
        results = {
            'method': method_name,
            'num_samples': len(all_metrics),
            'overall': {},
            'per_category': {}
        }
        
        # Overall metrics
        if len(all_metrics) > 0:
            metric_keys = all_metrics[0].keys()
            for key in metric_keys:
                if key != 'SSA':  # SSA is handled separately
                    values = [m[key] for m in all_metrics if key in m]
                    if len(values) > 0:
                        results['overall'][key] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values))
                        }
            
            # Aggregate SSA
            all_ssa = [m['SSA'] for m in all_metrics if 'SSA' in m]
            if len(all_ssa) > 0:
                species_scores = {}
                for ssa_dict in all_ssa:
                    for species, score in ssa_dict.items():
                        if species not in species_scores:
                            species_scores[species] = []
                        species_scores[species].append(score)
                
                results['overall']['SSA'] = {
                    species: {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores))
                    }
                    for species, scores in species_scores.items()
                }
        
        # Per-category metrics
        for category, category_metrics in per_category_metrics.items():
            if len(category_metrics) > 0:
                results['per_category'][category] = {}
                metric_keys = category_metrics[0].keys()
                
                for key in metric_keys:
                    if key != 'SSA':
                        values = [m[key] for m in category_metrics if key in m]
                        if len(values) > 0:
                            results['per_category'][category][key] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values))
                            }
        
        return results
    
    def _save_results(self, results: Dict, method_name: str):
        """Save results to JSON file"""
        output_file = os.path.join(
            self.results_dir, 
            f"{method_name}_results.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to: {output_file}")
    
    def _generate_visualizations(self, results: Dict, method_name: str):
        """Generate visualization plots"""
        
        # Plot 1: Overall metrics comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{method_name} - Evaluation Metrics', fontsize=16)
        
        overall_metrics = results['overall']
        
        # Extract standard metrics
        standard_metrics = ['mIoU', 'AP50', 'Coverage', 'CD', 'MOI', 'OIS', 'TCS', 'CAS']
        
        plot_idx = 0
        for metric_name in standard_metrics:
            if metric_name in overall_metrics and plot_idx < 6:
                ax = axes[plot_idx // 3, plot_idx % 3]
                
                data = overall_metrics[metric_name]
                mean = data['mean']
                std = data['std']
                
                ax.bar([method_name], [mean], yerr=[std], capsize=5)
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name}: {mean:.3f} ± {std:.3f}')
                ax.set_ylim([0, 1])
                
                plot_idx += 1
        
        # Remove unused subplots
        for i in range(plot_idx, 6):
            fig.delaxes(axes[i // 3, i % 3])
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f'{method_name}_metrics.png'),
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()
        
        # Plot 2: Per-category comparison
        if 'per_category' in results and len(results['per_category']) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            categories = list(results['per_category'].keys())
            metrics_to_plot = ['mIoU', 'MOI', 'TCS', 'CAS']
            
            x = np.arange(len(categories))
            width = 0.2
            
            for i, metric in enumerate(metrics_to_plot):
                means = []
                for cat in categories:
                    if metric in results['per_category'][cat]:
                        means.append(results['per_category'][cat][metric]['mean'])
                    else:
                        means.append(0)
                
                ax.bar(x + i * width, means, width, label=metric)
            
            ax.set_xlabel('Object Category')
            ax.set_ylabel('Score')
            ax.set_title('Performance by Object Category')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.results_dir, f'{method_name}_per_category.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
        
        print(f"📈 Visualizations saved to: {self.results_dir}/")
    
    def compare_methods(self, method_results: Dict[str, Dict]):
        """
        Compare multiple methods
        
        Args:
            method_results: Dict mapping method names to their results
        """
        print(f"\n{'='*60}")
        print("COMPARING METHODS")
        print(f"{'='*60}\n")
        
        # Create comparison table
        metrics_to_compare = ['mIoU', 'MOI', 'OIS', 'TCS', 'CAS', 'Coverage']
        
        comparison_table = []
        for method_name, results in method_results.items():
            row = [method_name]
            for metric in metrics_to_compare:
                if metric in results['overall']:
                    value = results['overall'][metric]['mean']
                    row.append(f"{value:.3f}")
                else:
                    row.append("N/A")
            comparison_table.append(row)
        
        # Print table
        header = ['Method'] + metrics_to_compare
        col_widths = [max(len(str(x)) for x in col) + 2 for col in zip(*([header] + comparison_table))]
        
        print("  ".join(h.ljust(w) for h, w in zip(header, col_widths)))
        print("-" * sum(col_widths) + "-" * (len(col_widths) - 1) * 2)
        
        for row in comparison_table:
            print("  ".join(str(x).ljust(w) for x, w in zip(row, col_widths)))
        
        # Create comparison visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        method_names = list(method_results.keys())
        x = np.arange(len(metrics_to_compare))
        width = 0.8 / len(method_names)
        
        for i, method_name in enumerate(method_names):
            results = method_results[method_name]
            means = []
            
            for metric in metrics_to_compare:
                if metric in results['overall']:
                    means.append(results['overall'][metric]['mean'])
                else:
                    means.append(0)
            
            ax.bar(x + i * width, means, width, label=method_name)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Method Comparison')
        ax.set_xticks(x + width * (len(method_names) - 1) / 2)
        ax.set_xticklabels(metrics_to_compare, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, 'method_comparison.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"\n📊 Comparison saved to: {self.results_dir}/method_comparison.png")


def main():
    """Example usage of evaluation pipeline"""
    
    # Initialize evaluator
    evaluator = MultiObjectEvaluator(
        test_dataset_path="./data/animal_kingdom/test",
        results_dir="./results"
    )
    
    # Evaluate your method
    your_results = evaluator.evaluate_method(
        method_name="LayeredSynthesis",
        generated_videos_dir="./outputs/layered_synthesis"
    )
    
    # Evaluate baseline (Peekaboo)
    baseline_results = evaluator.evaluate_method(
        method_name="Peekaboo_Baseline",
        generated_videos_dir="./outputs/peekaboo_baseline"
    )
    
    # Compare methods
    evaluator.compare_methods({
        'Layered Synthesis (Ours)': your_results,
        'Peekaboo Baseline': baseline_results
    })


if __name__ == "__main__":
    main()