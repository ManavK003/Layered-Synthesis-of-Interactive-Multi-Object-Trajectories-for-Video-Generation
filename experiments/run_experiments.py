"""
Experiment Runner for Multi-Object Video Generation
Runs comprehensive experiments and ablation studies
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
import sys
from typing import List, Dict
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generate_multi_object import MultiObjectVideoGenerator
from trajectory_controller import TrajectoryAwareController
from animal_motion_prior import AnimalMotionPrior


class ExperimentRunner:
    """
    Run systematic experiments for multi-object video generation
    """
    
    def __init__(
        self,
        output_dir: str = "./experiment_results",
        device: str = "cuda"
    ):
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
    def run_experiment_1_single_object(self):
        """
        Experiment 1: Single Object Comparison with Peekaboo
        Compare our method vs baseline on single objects
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: Single Object Comparison")
        print("="*60 + "\n")
        
        exp_dir = os.path.join(self.output_dir, "exp1_single_object")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Test cases
        test_cases = [
            {
                'prompt': 'A lion walking in the savanna',
                'objects': [{
                    'name': 'lion',
                    'species': 'lion',
                    'start_bbox': [0.2, 0.3, 0.4, 0.6],
                    'end_bbox': [0.6, 0.3, 0.8, 0.6],
                    'trajectory_type': 'smooth',
                    'priority': 0
                }]
            },
            {
                'prompt': 'An elephant walking through the forest',
                'objects': [{
                    'name': 'elephant',
                    'species': 'elephant',
                    'start_bbox': [0.1, 0.4, 0.3, 0.7],
                    'end_bbox': [0.7, 0.4, 0.9, 0.7],
                    'trajectory_type': 'smooth',
                    'priority': 0
                }]
            },
            {
                'prompt': 'A cheetah running fast',
                'objects': [{
                    'name': 'cheetah',
                    'species': 'cheetah',
                    'start_bbox': [0.1, 0.3, 0.25, 0.5],
                    'end_bbox': [0.75, 0.3, 0.9, 0.5],
                    'trajectory_type': 'accelerate',
                    'priority': 0
                }]
            }
        ]
        
        # Initialize generator
        generator = MultiObjectVideoGenerator(
            model_name="zeroscope",
            device=self.device,
            num_layers=4,
            enable_motion_priors=True
        )
        
        results = []
        
        for idx, test_case in enumerate(tqdm(test_cases, desc="Generating")):
            print(f"\nGenerating: {test_case['prompt']}")
            
            output_path = os.path.join(exp_dir, f"test_{idx:03d}.mp4")
            
            try:
                video = generator.generate_video(
                    prompt=test_case['prompt'],
                    objects=test_case['objects'],
                    num_frames=24,
                    num_inference_steps=50,
                    seed=42 + idx,
                    output_path=output_path
                )
                
                # Save trajectories for evaluation
                traj_controller = TrajectoryAwareController()
                trajectories = traj_controller.generate_trajectories(
                    objects=test_case['objects'],
                    num_frames=24
                )
                
                pred_file = os.path.join(exp_dir, f"test_{idx:03d}_predictions.json")
                with open(pred_file, 'w') as f:
                    json.dump({
                        'trajectories': [traj.tolist() for traj in trajectories.values()],
                        'prompt': test_case['prompt'],
                        'objects': test_case['objects']
                    }, f, indent=2)
                
                results.append({
                    'test_id': idx,
                    'status': 'success',
                    'output': output_path
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'test_id': idx,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save experiment summary
        summary_file = os.path.join(exp_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment': 'Single Object Comparison',
                'num_tests': len(test_cases),
                'results': results
            }, f, indent=2)
        
        print(f"\n✅ Experiment 1 complete! Results in: {exp_dir}")
    
    def run_experiment_2_two_objects(self):
        """
        Experiment 2: Two-Object Interactions
        Test object interactions and collision avoidance
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Two-Object Interactions")
        print("="*60 + "\n")
        
        exp_dir = os.path.join(self.output_dir, "exp2_two_objects")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Test cases with two objects
        test_cases = [
            {
                'prompt': 'A lion chasing a zebra in the savanna',
                'objects': [
                    {
                        'name': 'lion',
                        'species': 'lion',
                        'start_bbox': [0.1, 0.3, 0.3, 0.6],
                        'end_bbox': [0.6, 0.3, 0.8, 0.6],
                        'trajectory_type': 'accelerate',
                        'priority': 0
                    },
                    {
                        'name': 'zebra',
                        'species': 'zebra',
                        'start_bbox': [0.6, 0.4, 0.8, 0.7],
                        'end_bbox': [0.2, 0.2, 0.4, 0.5],
                        'trajectory_type': 'accelerate',
                        'priority': 1
                    }
                ]
            },
            {
                'prompt': 'Two elephants walking together',
                'objects': [
                    {
                        'name': 'elephant1',
                        'species': 'elephant',
                        'start_bbox': [0.1, 0.3, 0.3, 0.7],
                        'end_bbox': [0.6, 0.3, 0.8, 0.7],
                        'trajectory_type': 'smooth',
                        'priority': 0
                    },
                    {
                        'name': 'elephant2',
                        'species': 'elephant',
                        'start_bbox': [0.15, 0.35, 0.35, 0.75],
                        'end_bbox': [0.65, 0.35, 0.85, 0.75],
                        'trajectory_type': 'smooth',
                        'priority': 0
                    }
                ]
            },
            {
                'prompt': 'A leopard and a gazelle in the grassland',
                'objects': [
                    {
                        'name': 'leopard',
                        'species': 'leopard',
                        'start_bbox': [0.1, 0.4, 0.3, 0.7],
                        'end_bbox': [0.5, 0.4, 0.7, 0.7],
                        'trajectory_type': 'smooth',
                        'priority': 0
                    },
                    {
                        'name': 'gazelle',
                        'species': 'gazelle',
                        'start_bbox': [0.6, 0.3, 0.75, 0.5],
                        'end_bbox': [0.7, 0.5, 0.85, 0.7],
                        'trajectory_type': 'linear',
                        'priority': 1
                    }
                ]
            }
        ]
        
        generator = MultiObjectVideoGenerator(
            model_name="zeroscope",
            device=self.device,
            num_layers=4,
            enable_motion_priors=True
        )
        
        results = []
        
        for idx, test_case in enumerate(tqdm(test_cases, desc="Generating")):
            print(f"\nGenerating: {test_case['prompt']}")
            
            output_path = os.path.join(exp_dir, f"test_{idx:03d}.mp4")
            
            try:
                video = generator.generate_video(
                    prompt=test_case['prompt'],
                    objects=test_case['objects'],
                    num_frames=24,
                    num_inference_steps=50,
                    seed=42 + idx,
                    output_path=output_path
                )
                
                # Save predictions
                traj_controller = TrajectoryAwareController()
                trajectories = traj_controller.generate_trajectories(
                    objects=test_case['objects'],
                    num_frames=24,
                    resolve_collisions=True
                )
                
                pred_file = os.path.join(exp_dir, f"test_{idx:03d}_predictions.json")
                with open(pred_file, 'w') as f:
                    json.dump({
                        'trajectories': [traj.tolist() for traj in trajectories.values()],
                        'prompt': test_case['prompt'],
                        'objects': test_case['objects']
                    }, f, indent=2)
                
                results.append({
                    'test_id': idx,
                    'status': 'success',
                    'output': output_path
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'test_id': idx,
                    'status': 'failed',
                    'error': str(e)
                })
        
        summary_file = os.path.join(exp_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment': 'Two-Object Interactions',
                'num_tests': len(test_cases),
                'results': results
            }, f, indent=2)
        
        print(f"\n✅ Experiment 2 complete! Results in: {exp_dir}")
    
    def run_experiment_3_multi_objects(self):
        """
        Experiment 3: Multi-Object Complex Scenes (3+ objects)
        Test with complex multi-animal scenarios
        """
        print("\n" + "="*60)
        print("EXPERIMENT 3: Multi-Object Complex Scenes")
        print("="*60 + "\n")
        
        exp_dir = os.path.join(self.output_dir, "exp3_multi_objects")
        os.makedirs(exp_dir, exist_ok=True)
        
        test_cases = [
            {
                'prompt': 'A pride of lions resting in the savanna',
                'objects': [
                    {
                        'name': 'lion1',
                        'species': 'lion',
                        'start_bbox': [0.1, 0.3, 0.25, 0.5],
                        'end_bbox': [0.15, 0.35, 0.3, 0.55],
                        'trajectory_type': 'smooth',
                        'priority': 0
                    },
                    {
                        'name': 'lion2',
                        'species': 'lion',
                        'start_bbox': [0.3, 0.4, 0.45, 0.6],
                        'end_bbox': [0.35, 0.4, 0.5, 0.6],
                        'trajectory_type': 'smooth',
                        'priority': 0
                    },
                    {
                        'name': 'lion3',
                        'species': 'lion',
                        'start_bbox': [0.5, 0.3, 0.65, 0.5],
                        'end_bbox': [0.55, 0.35, 0.7, 0.55],
                        'trajectory_type': 'smooth',
                        'priority': 1
                    }
                ]
            },
            {
                'prompt': 'A herd of zebras migrating together',
                'objects': [
                    {
                        'name': f'zebra{i}',
                        'species': 'zebra',
                        'start_bbox': [0.1 + i*0.15, 0.3 + (i%2)*0.1, 0.2 + i*0.15, 0.5 + (i%2)*0.1],
                        'end_bbox': [0.4 + i*0.15, 0.4 + (i%2)*0.1, 0.5 + i*0.15, 0.6 + (i%2)*0.1],
                        'trajectory_type': 'smooth',
                        'priority': i % 2
                    }
                    for i in range(4)
                ]
            }
        ]
        
        generator = MultiObjectVideoGenerator(
            model_name="zeroscope",
            device=self.device,
            num_layers=4,
            enable_motion_priors=True
        )
        
        results = []
        
        for idx, test_case in enumerate(tqdm(test_cases, desc="Generating")):
            print(f"\nGenerating: {test_case['prompt']}")
            print(f"Number of objects: {len(test_case['objects'])}")
            
            output_path = os.path.join(exp_dir, f"test_{idx:03d}.mp4")
            
            try:
                video = generator.generate_video(
                    prompt=test_case['prompt'],
                    objects=test_case['objects'],
                    num_frames=24,
                    num_inference_steps=50,
                    seed=42 + idx,
                    output_path=output_path
                )
                
                # Save predictions
                traj_controller = TrajectoryAwareController()
                trajectories = traj_controller.generate_trajectories(
                    objects=test_case['objects'],
                    num_frames=24,
                    resolve_collisions=True
                )
                
                pred_file = os.path.join(exp_dir, f"test_{idx:03d}_predictions.json")
                with open(pred_file, 'w') as f:
                    json.dump({
                        'trajectories': [traj.tolist() for traj in trajectories.values()],
                        'prompt': test_case['prompt'],
                        'objects': test_case['objects']
                    }, f, indent=2)
                
                results.append({
                    'test_id': idx,
                    'status': 'success',
                    'output': output_path,
                    'num_objects': len(test_case['objects'])
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'test_id': idx,
                    'status': 'failed',
                    'error': str(e)
                })
        
        summary_file = os.path.join(exp_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment': 'Multi-Object Complex Scenes',
                'num_tests': len(test_cases),
                'results': results
            }, f, indent=2)
        
        print(f"\n✅ Experiment 3 complete! Results in: {exp_dir}")
    
    def run_experiment_4_ablation(self):
        """
        Experiment 4: Ablation Study
        Test each component separately
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: Ablation Study")
        print("="*60 + "\n")
        
        exp_dir = os.path.join(self.output_dir, "exp4_ablation")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Test case for ablation
        test_case = {
            'prompt': 'A lion and a zebra in the savanna',
            'objects': [
                {
                    'name': 'lion',
                    'species': 'lion',
                    'start_bbox': [0.1, 0.3, 0.3, 0.6],
                    'end_bbox': [0.6, 0.3, 0.8, 0.6],
                    'trajectory_type': 'smooth',
                    'priority': 0
                },
                {
                    'name': 'zebra',
                    'species': 'zebra',
                    'start_bbox': [0.6, 0.4, 0.8, 0.7],
                    'end_bbox': [0.2, 0.4, 0.4, 0.7],
                    'trajectory_type': 'smooth',
                    'priority': 1
                }
            ]
        }
        
        ablation_configs = [
            {
                'name': 'full_model',
                'description': 'Full model with all components',
                'num_layers': 4,
                'enable_motion_priors': True,
                'resolve_collisions': True
            },
            {
                'name': 'no_layering',
                'description': 'Without layered attention (single layer)',
                'num_layers': 1,
                'enable_motion_priors': True,
                'resolve_collisions': True
            },
            {
                'name': 'no_motion_priors',
                'description': 'Without animal motion priors',
                'num_layers': 4,
                'enable_motion_priors': False,
                'resolve_collisions': True
            },
            {
                'name': 'no_collision_resolution',
                'description': 'Without collision resolution',
                'num_layers': 4,
                'enable_motion_priors': True,
                'resolve_collisions': False
            }
        ]
        
        results = []
        
        for config in tqdm(ablation_configs, desc="Running ablations"):
            print(f"\nRunning: {config['description']}")
            
            config_dir = os.path.join(exp_dir, config['name'])
            os.makedirs(config_dir, exist_ok=True)
            
            try:
                # Initialize generator with ablation config
                generator = MultiObjectVideoGenerator(
                    model_name="zeroscope",
                    device=self.device,
                    num_layers=config['num_layers'],
                    enable_motion_priors=config['enable_motion_priors']
                )
                
                output_path = os.path.join(config_dir, "video.mp4")
                
                video = generator.generate_video(
                    prompt=test_case['prompt'],
                    objects=test_case['objects'],
                    num_frames=24,
                    num_inference_steps=50,
                    seed=42,
                    output_path=output_path
                )
                
                # Generate trajectories with ablation config
                traj_controller = TrajectoryAwareController()
                trajectories = traj_controller.generate_trajectories(
                    objects=test_case['objects'],
                    num_frames=24,
                    resolve_collisions=config['resolve_collisions']
                )
                
                # Compute trajectory metrics
                metrics = traj_controller.compute_trajectory_metrics(trajectories)
                
                pred_file = os.path.join(config_dir, "predictions.json")
                with open(pred_file, 'w') as f:
                    json.dump({
                        'trajectories': [traj.tolist() for traj in trajectories.values()],
                        'trajectory_metrics': metrics,
                        'config': config
                    }, f, indent=2)
                
                results.append({
                    'config': config['name'],
                    'description': config['description'],
                    'status': 'success',
                    'output': output_path,
                    'trajectory_metrics': metrics
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'config': config['name'],
                    'description': config['description'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        summary_file = os.path.join(exp_dir, "ablation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment': 'Ablation Study',
                'test_case': test_case,
                'configurations': ablation_configs,
                'results': results
            }, f, indent=2)
        
        print(f"\n✅ Experiment 4 complete! Results in: {exp_dir}")
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print("="*60 + "\n")
        
        self.run_experiment_1_single_object()
        self.run_experiment_2_two_objects()
        self.run_experiment_3_multi_objects()
        self.run_experiment_4_ablation()
        
        print("\n" + "="*60)
        print("🎉 ALL EXPERIMENTS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Object Video Generation Experiments")
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment", type=str, choices=['1', '2', '3', '4', 'all'], default='all',
                        help="Which experiment to run (1=single, 2=two-obj, 3=multi-obj, 4=ablation, all=all)")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        device=args.device
    )
    
    if args.experiment == '1':
        runner.run_experiment_1_single_object()
    elif args.experiment == '2':
        runner.run_experiment_2_two_objects()
    elif args.experiment == '3':
        runner.run_experiment_3_multi_objects()
    elif args.experiment == '4':
        runner.run_experiment_4_ablation()
    else:
        runner.run_all_experiments()


if __name__ == "__main__":
    main()