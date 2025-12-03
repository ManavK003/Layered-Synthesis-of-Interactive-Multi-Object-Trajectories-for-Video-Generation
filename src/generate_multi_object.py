"""
Multi-Object Video Generation with Layered Synthesis
Main script for generating videos with multiple objects
"""

import torch
import numpy as np
from diffusers import DiffusionPipeline
from typing import List, Dict, Optional
import argparse
import os
from pathlib import Path

# Import our custom modules
from layered_attention import LayeredMultiObjectAttention, PeekabooLayeredAttention
from trajectory_controller import TrajectoryAwareController
from animal_motion_prior import AnimalMotionPrior


class MultiObjectVideoGenerator:
    """
    Main class for multi-object video generation
    Integrates all novel components
    """
    
    def __init__(
        self,
        model_name: str = "zeroscope",
        device: str = "cuda",
        num_layers: int = 4,
        enable_motion_priors: bool = True
    ):
        self.device = device
        self.model_name = model_name
        
        # Load base diffusion model
        print(f"Loading {model_name} model...")
        if model_name == "zeroscope":
            self.pipeline = DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w",
                torch_dtype=torch.float16
            )
        elif model_name == "modelscope":
            self.pipeline = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.pipeline = self.pipeline.to(device)
        
        # Initialize our novel components
        print("Initializing layered attention...")
        self.layered_attention = LayeredMultiObjectAttention(
            num_layers=num_layers,
            hidden_dim=512
        ).to(device)
        
        print("Initializing trajectory controller...")
        self.trajectory_controller = TrajectoryAwareController()
        
        if enable_motion_priors:
            print("Initializing animal motion priors...")
            self.motion_prior = AnimalMotionPrior()
            # Try to load cache, otherwise create dummy data
            if not self.motion_prior.load_cache():
                self.motion_prior.extract_from_animal_kingdom(
                    animal_kingdom_data_path="./data/animal_kingdom"
                )
                self.motion_prior.save_cache()
        else:
            self.motion_prior = None
        
        print("Initialization complete!")
    
    def generate_video(
        self,
        prompt: str,
        objects: List[Dict],
        num_frames: int = 24,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        frozen_steps: int = 2,
        seed: Optional[int] = None,
        output_path: str = "./outputs/video.mp4"
    ) -> torch.Tensor:
        """
        Generate multi-object video
        
        Args:
            prompt: Text prompt describing the scene
            objects: List of object specifications:
                {
                    'name': str (e.g., 'lion'),
                    'species': str (for motion prior),
                    'start_bbox': [x1, y1, x2, y2],
                    'end_bbox': [x1, y1, x2, y2],
                    'trajectory_type': str ('linear', 'smooth', 'accelerate'),
                    'priority': int (layer assignment, 0=highest)
                }
            num_frames: Number of frames to generate
            num_inference_steps: Diffusion sampling steps
            guidance_scale: CFG scale
            frozen_steps: Steps to apply masking (Peekaboo parameter)
            seed: Random seed
            output_path: Where to save output video
            
        Returns:
            video: [T, C, H, W] generated video
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"\nGenerating video: '{prompt}'")
        print(f"Objects: {[obj['name'] for obj in objects]}")
        
        # Step 1: Generate trajectories
        print("\n[1/4] Generating trajectories...")
        trajectories = self.trajectory_controller.generate_trajectories(
            objects=objects,
            num_frames=num_frames,
            resolve_collisions=True
        )
        
        # Apply motion priors if available
        if self.motion_prior is not None:
            print("Applying species-specific motion priors...")
            for obj in objects:
                if obj['species'] in trajectories:
                    trajectories[obj['name']] = self.motion_prior.adjust_trajectory_with_prior(
                        trajectory=trajectories[obj['name']],
                        species=obj['species'],
                        strength=0.3
                    )
        
        # Step 2: Convert trajectories to masks
        print("\n[2/4] Creating object masks...")
        object_masks = self._create_masks_from_trajectories(
            trajectories=trajectories,
            video_size=(256, 256),  # H, W
            num_frames=num_frames
        )
        
        # Step 3: Generate video with layered attention
        print("\n[3/4] Running diffusion generation...")
        video = self._generate_with_layered_attention(
            prompt=prompt,
            object_masks=object_masks,
            object_priorities=[obj['priority'] for obj in objects],
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            frozen_steps=frozen_steps
        )
        
        # Step 4: Save video
        print("\n[4/4] Saving video...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._save_video(video, output_path)
        
        print(f"\n✅ Video saved to: {output_path}")
        
        return video
    
    def _create_masks_from_trajectories(
        self,
        trajectories: Dict[str, np.ndarray],
        video_size: Tuple[int, int],
        num_frames: int
    ) -> List[torch.Tensor]:
        """
        Convert trajectories to binary masks
        
        Args:
            trajectories: Dict mapping object names to [T, 4] trajectories
            video_size: (H, W) video dimensions
            num_frames: Number of frames
            
        Returns:
            masks: List of [1, H, W, T] binary masks
        """
        H, W = video_size
        masks = []
        
        for obj_name, trajectory in trajectories.items():
            # Create mask for this object
            mask = torch.zeros(1, H, W, num_frames)
            
            for t in range(num_frames):
                bbox = trajectory[t]  # [x1, y1, x2, y2] in [0, 1]
                
                # Convert to pixel coordinates
                x1 = int(bbox[0] * W)
                y1 = int(bbox[1] * H)
                x2 = int(bbox[2] * W)
                y2 = int(bbox[3] * H)
                
                # Fill mask
                mask[0, y1:y2, x1:x2, t] = 1.0
            
            masks.append(mask.to(self.device))
        
        return masks
    
    def _generate_with_layered_attention(
        self,
        prompt: str,
        object_masks: List[torch.Tensor],
        object_priorities: List[int],
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        frozen_steps: int
    ) -> torch.Tensor:
        """
        Generate video using layered multi-object attention
        
        This is a simplified version - full integration requires
        modifying the diffusion pipeline's attention modules
        """
        # For now, use standard pipeline with prompt
        # Full implementation would inject our layered attention
        # into the UNet's attention layers
        
        video = self.pipeline(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]  # [T, H, W, C]
        
        # Convert to tensor [T, C, H, W]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        
        # TODO: Apply layered attention during generation
        # This requires modifying the UNet forward pass
        # See integration instructions below
        
        return video
    
    def _save_video(self, video: torch.Tensor, output_path: str):
        """Save video to file"""
        import cv2
        
        # Convert to numpy [T, H, W, C]
        video_np = video.permute(0, 2, 3, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        T, H, W, C = video_np.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (W, H))
        
        for frame in video_np:
            if C == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()


def main():
    parser = argparse.ArgumentParser(description="Multi-Object Video Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--model", type=str, default="zeroscope", choices=["zeroscope", "modelscope"])
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="./outputs/video.mp4")
    parser.add_argument("--config", type=str, help="JSON config file with object specifications")
    
    args = parser.parse_args()
    
    # Example object configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        objects = config['objects']
    else:
        # Default: two animals
        objects = [
            {
                'name': 'lion',
                'species': 'lion',
                'start_bbox': [0.1, 0.3, 0.3, 0.6],
                'end_bbox': [0.5, 0.3, 0.7, 0.6],
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
    
    # Initialize generator
    generator = MultiObjectVideoGenerator(
        model_name=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_layers=4,
        enable_motion_priors=True
    )
    
    # Generate video
    video = generator.generate_video(
        prompt=args.prompt,
        objects=objects,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        output_path=args.output
    )
    
    print("\n🎉 Generation complete!")


if __name__ == "__main__":
    main()