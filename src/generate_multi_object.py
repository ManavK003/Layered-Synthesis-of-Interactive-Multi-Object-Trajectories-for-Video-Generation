import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import warnings
import cv2
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torchvision.io as vision_io

from models.pipelines import TextToVideoSDPipelineSpatialAware
from diffusers.utils import export_to_video
from PIL import Image
import torchvision

import argparse

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description="Generate videos with different prompts and fg objects")
    parser.add_argument("--model", type=str, default="zeroscope", choices=["zeroscope", "modelscope"], help="Model to use for the generation")
    parser.add_argument("--prompt", type=str, default="A panda eating bamboo in a lush bamboo forest.", help="Prompt to generate the video")
    parser.add_argument("--fg_object", type=str, default="panda", help="Foreground objects (comma-separated for multiple)")
    parser.add_argument("--frozen_steps", type=int, default=2, help="Number of frozen steps")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (default: 16 for modelscope, 24 for zeroscope)")
    parser.add_argument("--seed", type=int, default=2, help="Seed for random number generation")
    parser.add_argument("--output_path", type=str, default="src/demo", help="Path to save the generated videos")
    return parser

def create_multi_object_masks(num_frames, height_latent, width_latent, num_objects, device, dtype):
    """
    Create separate bounding box masks for multiple objects.
    Objects will be positioned in different areas and move differently.
    """
    # Create base mask tensor
    bbox_mask = torch.zeros([num_frames, 1, height_latent, width_latent], device=device).to(dtype)
    
    if num_objects == 1:
        # Single object: centered, slight movement
        box_size = max(4, height_latent // 8)  # Adaptive size based on resolution
        x_start = [height_latent//2 - box_size//2 + (i % 2) for i in range(num_frames)]
        x_end = [height_latent//2 + box_size//2 + (i % 2) for i in range(num_frames)]
        y_start = [width_latent//2 - box_size//2 for _ in range(num_frames)]
        y_end = [width_latent//2 + box_size//2 for _ in range(num_frames)]
        
        for i in range(num_frames):
            bbox_mask[i, :, x_start[i]:x_end[i], y_start[i]:y_end[i]] = 1
            
    elif num_objects == 2:
        # Smaller boxes for multi-object
        box_size = max(3, height_latent // 10)  # Smaller for multiple objects
        
        # Object 1: Left side, moving right
        x1_start = [height_latent//4 - box_size//2 + (i // 3) for i in range(num_frames)]
        x1_end = [height_latent//4 + box_size//2 + (i // 3) for i in range(num_frames)]
        y1_start = [width_latent//2 - box_size//2 for _ in range(num_frames)]
        y1_end = [width_latent//2 + box_size//2 for _ in range(num_frames)]
        
        # Object 2: Right side, moving left
        x2_start = [3*height_latent//4 - box_size//2 - (i // 3) for i in range(num_frames)]
        x2_end = [3*height_latent//4 + box_size//2 - (i // 3) for i in range(num_frames)]
        y2_start = [width_latent//2 - box_size//2 for _ in range(num_frames)]
        y2_end = [width_latent//2 + box_size//2 for _ in range(num_frames)]
        
        # Add both objects to mask
        for i in range(num_frames):
            # Ensure boundaries are valid
            x1_s = max(0, min(x1_start[i], height_latent - 1))
            x1_e = max(0, min(x1_end[i], height_latent))
            x2_s = max(0, min(x2_start[i], height_latent - 1))
            x2_e = max(0, min(x2_end[i], height_latent))
            
            y1_s = max(0, min(y1_start[i], width_latent - 1))
            y1_e = max(0, min(y1_end[i], width_latent))
            y2_s = max(0, min(y2_start[i], width_latent - 1))
            y2_e = max(0, min(y2_end[i], width_latent))
            
            bbox_mask[i, :, x1_s:x1_e, y1_s:y1_e] = 1  # Object 1
            bbox_mask[i, :, x2_s:x2_e, y2_s:y2_e] = 1  # Object 2
            
    else:
        # More than 2 objects: distribute across the frame
        box_size = max(3, height_latent // 12)  # Even smaller for 3+ objects
        
        for obj_idx in range(num_objects):
            # Distribute objects horizontally
            base_x = int((obj_idx + 1) * height_latent / (num_objects + 1))
            base_y = int(width_latent / 2)
            
            x_start = [base_x - box_size//2 + (i % 2) for i in range(num_frames)]
            x_end = [base_x + box_size//2 + (i % 2) for i in range(num_frames)]
            y_start = [base_y - box_size//2 for _ in range(num_frames)]
            y_end = [base_y + box_size//2 for _ in range(num_frames)]
            
            for i in range(num_frames):
                x_s = max(0, min(x_start[i], height_latent - 1))
                x_e = max(0, min(x_end[i], height_latent))
                y_s = max(0, min(y_start[i], width_latent - 1))
                y_e = max(0, min(y_end[i], width_latent))
                
                bbox_mask[i, :, x_s:x_e, y_s:y_e] = 1
    
    return bbox_mask

def generate_video(pipe, overall_prompt, latents, get_latents=False, num_frames=24, num_inference_steps=50, fg_masks=None, 
        fg_masked_latents=None, frozen_steps=0, custom_attention_mask=None, fg_prompt=None, height=320, width=576):
    
    video_frames = pipe(overall_prompt, num_frames=num_frames, latents=latents, num_inference_steps=num_inference_steps, frozen_mask=fg_masks, 
    frozen_steps=frozen_steps, latents_all_input=fg_masked_latents, custom_attention_mask=custom_attention_mask, fg_prompt=fg_prompt,
    make_attention_mask_2d=True, attention_mask_block_diagonal=True, height=height, width=width).frames
    if get_latents:
        video_latents = pipe(overall_prompt, num_frames=num_frames, latents=latents, num_inference_steps=num_inference_steps, output_type="latent").frames
        return video_frames, video_latents
    
    return video_frames

def save_frames(path):
    video, audio, video_info = vision_io.read_video(f"{path}.mp4", pts_unit='sec')
    num_frames = video.size(0)
    os.makedirs(f"{path}", exist_ok=True)
    for i in range(num_frames):
        frame = video[i, :, :, :].numpy()
        img = Image.fromarray(frame.astype('uint8'))
        img.save(f"{path}/frame_{i:04d}.png")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.frozen_steps <= args.num_inference_steps, "Frozen steps should be less than or equal to the number of inference steps"
    
    # Parse multiple objects
    fg_objects = [obj.strip() for obj in args.fg_object.split(',')]
    num_objects = len(fg_objects)
    
    print(f"Generating video with {num_objects} object(s): {fg_objects}")
    
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "zeroscope":
        pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
            "cerspense/zeroscope_v2_576w", torch_dtype=torch.float16).to(torch_device)
        num_frames = args.num_frames if args.num_frames else 24  # Default 24 for zeroscope
        random_latents = torch.randn([1, 4, num_frames, 40, 72], generator=torch.Generator().manual_seed(args.seed)).to(torch_device).to(torch.float16)
        height = 320
        width = 576
        height_latent = 40
        width_latent = 72
    elif args.model == "modelscope":
        pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", variant="fp16").to(torch.float16).to(torch_device)
        num_frames = args.num_frames if args.num_frames else 16  # Default 16 for modelscope
        random_latents = torch.randn([1, 4, num_frames, 32, 32], generator=torch.Generator().manual_seed(args.seed)).to(torch_device).to(torch.float16)
        height = 256
        width = 256
        height_latent = 32
        width_latent = 32

    # Create multi-object masks
    fg_masks = create_multi_object_masks(num_frames, height_latent, width_latent, num_objects, torch_device, torch.float16)
    
    fg_masked_latents = None
    fg_object_str = args.fg_object
    overall_prompt = args.prompt

    save_path = args.model
    os.makedirs(f"{args.output_path}/{save_path}/{overall_prompt}-mask", exist_ok=True)
    
    # Save mask visualizations
    for i in range(num_frames):
        torchvision.utils.save_image(fg_masks[i], f"{args.output_path}/{save_path}/{overall_prompt}-mask/frame_{i:04d}.png")

    print(f"Generating video for prompt: {overall_prompt}")
    print(f"Foreground objects: {fg_object_str}")
    print(f"Number of objects: {num_objects}")
    
    video_frames = generate_video(pipe, overall_prompt, random_latents, get_latents=False, num_frames=num_frames, num_inference_steps=args.num_inference_steps, 
        fg_masks=fg_masks, fg_masked_latents=fg_masked_latents, frozen_steps=args.frozen_steps, fg_prompt=fg_object_str, height=height, width=width)
    
    # Save video frames
    overall_prompt_clean = overall_prompt.replace(" ", "_")
    os.makedirs(f"{args.output_path}/{save_path}/{overall_prompt_clean}", exist_ok=True)
    video_path = export_to_video(video_frames, f"{args.output_path}/{save_path}/{overall_prompt_clean}/{args.frozen_steps}_of_{args.num_inference_steps}_{args.seed}_peekaboo_multi.mp4")
    save_frames(f"{args.output_path}/{save_path}/{overall_prompt_clean}/{args.frozen_steps}_of_{args.num_inference_steps}_{args.seed}_peekaboo_multi")
    print(f"Video saved at {video_path}")
    print(f"Masks saved at {args.output_path}/{save_path}/{overall_prompt}-mask/")