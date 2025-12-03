"""
Bounding box utility functions
"""

import numpy as np
import torch
from typing import List, Tuple


def bbox_to_mask(
    bbox: np.ndarray,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Convert bounding box to binary mask
    
    Args:
        bbox: [4] bounding box [x1, y1, x2, y2] in [0, 1]
        image_size: (H, W) image dimensions
        
    Returns:
        mask: [H, W] binary mask
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    x1 = int(bbox[0] * W)
    y1 = int(bbox[1] * H)
    x2 = int(bbox[2] * W)
    y2 = int(bbox[3] * H)
    
    mask[y1:y2, x1:x2] = 1.0
    
    return mask


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Convert binary mask to bounding box
    
    Args:
        mask: [H, W] binary mask
        
    Returns:
        bbox: [4] bounding box [x1, y1, x2, y2] in [0, 1]
    """
    H, W = mask.shape
    
    # Find non-zero pixels
    rows, cols = np.where(mask > 0.5)
    
    if len(rows) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    y1 = rows.min() / H
    y2 = rows.max() / H
    x1 = cols.min() / W
    x2 = cols.max() / W
    
    return np.array([x1, y1, x2, y2])


def compute_bbox_area(bbox: np.ndarray) -> float:
    """Compute bounding box area"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return float(width * height)


def is_bbox_valid(
    bbox: np.ndarray,
    min_size: float = 0.01
) -> bool:
    """Check if bounding box is valid"""
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return False
    area = compute_bbox_area(bbox)
    return area >= min_size