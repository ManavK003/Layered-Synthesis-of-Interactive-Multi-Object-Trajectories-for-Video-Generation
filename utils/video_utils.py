"""
Video utility functions
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple
import imageio


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 8
):
    """
    Save video frames to file
    
    Args:
        frames: [T, C, H, W] or [T, H, W, C] tensor
        output_path: Output file path
        fps: Frames per second
    """
    # Convert to numpy
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # Ensure [T, H, W, C]
    if frames.shape[1] == 3 or frames.shape[1] == 1:
        frames = frames.transpose(0, 2, 3, 1)
    
    # Normalize to [0, 255]
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    
    # Save with imageio
    imageio.mimsave(output_path, frames, fps=fps)


def visualize_trajectory(
    frames: np.ndarray,
    trajectory: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw trajectory on video frames
    
    Args:
        frames: [T, H, W, C] video frames
        trajectory: [T, 4] bounding box trajectory
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        frames_with_traj: Frames with drawn trajectory
    """
    frames_out = frames.copy()
    T, H, W, C = frames.shape
    
    for t in range(T):
        bbox = trajectory[t]
        x1, y1, x2, y2 = bbox
        
        # Convert to pixel coordinates
        x1_px = int(x1 * W)
        y1_px = int(y1 * H)
        x2_px = int(x2 * W)
        y2_px = int(y2 * H)
        
        # Draw bbox
        cv2.rectangle(
            frames_out[t],
            (x1_px, y1_px),
            (x2_px, y2_px),
            color,
            thickness
        )
        
        # Draw trajectory line
        if t > 0:
            prev_bbox = trajectory[t-1]
            prev_cx = int(((prev_bbox[0] + prev_bbox[2]) / 2) * W)
            prev_cy = int(((prev_bbox[1] + prev_bbox[3]) / 2) * H)
            curr_cx = int(((bbox[0] + bbox[2]) / 2) * W)
            curr_cy = int(((bbox[1] + bbox[3]) / 2) * H)
            
            cv2.line(
                frames_out[t],
                (prev_cx, prev_cy),
                (curr_cx, curr_cy),
                color,
                thickness
            )
    
    return frames_out


def create_side_by_side_comparison(
    video1: np.ndarray,
    video2: np.ndarray,
    labels: List[str] = ["Method 1", "Method 2"]
) -> np.ndarray:
    """
    Create side-by-side video comparison
    
    Args:
        video1: [T, H, W, C] first video
        video2: [T, H, W, C] second video
        labels: Labels for each video
        
    Returns:
        comparison: [T, H, W*2, C] side-by-side video
    """
    T, H, W, C = video1.shape
    
    # Create canvas
    comparison = np.zeros((T, H, W*2, C), dtype=video1.dtype)
    
    # Place videos side by side
    comparison[:, :, :W, :] = video1
    comparison[:, :, W:, :] = video2
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(T):
        cv2.putText(
            comparison[t],
            labels[0],
            (10, 30),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            comparison[t],
            labels[1],
            (W + 10, 30),
            font,
            1,
            (255, 255, 255),
            2
        )
    
    return comparison