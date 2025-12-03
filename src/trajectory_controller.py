"""
Trajectory-Aware Controller for Multi-Object Video Generation
Ensures physically plausible multi-object trajectories
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d


class CollisionDetector:
    """Detects and resolves collisions between objects"""
    
    def __init__(self, collision_threshold: float = 0.1):
        self.collision_threshold = collision_threshold
        
    def detect_collisions(
        self, 
        bboxes: List[torch.Tensor]
    ) -> List[Tuple[int, int]]:
        """
        Detect overlapping bounding boxes
        
        Args:
            bboxes: List of [x1, y1, x2, y2] bounding boxes
            
        Returns:
            collisions: List of (obj1_idx, obj2_idx) collision pairs
        """
        collisions = []
        num_objects = len(bboxes)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                iou = self.compute_iou(bboxes[i], bboxes[j])
                if iou > self.collision_threshold:
                    collisions.append((i, j))
        
        return collisions
    
    def compute_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Intersection area
        if x2_i < x1_i or y2_i < y1_i:
            intersection = 0.0
        else:
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        return float(iou)
    
    def resolve(self, bboxes: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Resolve collisions by slightly adjusting positions
        
        Args:
            bboxes: List of bounding boxes
            
        Returns:
            adjusted_bboxes: Collision-free bounding boxes
        """
        adjusted = [bbox.clone() for bbox in bboxes]
        collisions = self.detect_collisions(adjusted)
        
        # Iteratively resolve collisions
        max_iterations = 10
        iteration = 0
        
        while collisions and iteration < max_iterations:
            for i, j in collisions:
                # Move boxes apart slightly
                center_i = (adjusted[i][:2] + adjusted[i][2:]) / 2
                center_j = (adjusted[j][:2] + adjusted[j][2:]) / 2
                
                direction = center_i - center_j
                direction = direction / (torch.norm(direction) + 1e-6)
                
                # Move objects apart
                offset = direction * 0.05
                adjusted[i][:2] += offset
                adjusted[i][2:] += offset
                adjusted[j][:2] -= offset
                adjusted[j][2:] -= offset
                
                # Clamp to valid range [0, 1]
                adjusted[i] = torch.clamp(adjusted[i], 0.0, 1.0)
                adjusted[j] = torch.clamp(adjusted[j], 0.0, 1.0)
            
            collisions = self.detect_collisions(adjusted)
            iteration += 1
        
        return adjusted


class MotionPredictor(nn.Module):
    """Predicts next frame positions based on motion history"""
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM for temporal motion modeling
        self.lstm = nn.LSTM(
            input_size=4,  # (x1, y1, x2, y2)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Predict next bbox
        )
        
    def forward(
        self, 
        trajectory_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next bounding box position
        
        Args:
            trajectory_history: [B, T_hist, 4] past bounding boxes
            
        Returns:
            next_bbox: [B, 4] predicted next bounding box
        """
        lstm_out, _ = self.lstm(trajectory_history)
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_dim]
        next_bbox = self.output(last_hidden)
        
        return next_bbox


class TrajectoryAwareController:
    """
    Main controller for multi-object trajectory generation
    Ensures smooth, collision-free, physically plausible trajectories
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        smoothing_window: int = 5
    ):
        self.image_size = image_size
        self.smoothing_window = smoothing_window
        self.collision_detector = CollisionDetector()
        self.motion_predictor = MotionPredictor()
        
    def interpolate_trajectory(
        self,
        start_bbox: np.ndarray,
        end_bbox: np.ndarray,
        num_frames: int,
        trajectory_type: str = 'linear'
    ) -> np.ndarray:
        """
        Interpolate trajectory between start and end positions
        
        Args:
            start_bbox: [4] starting bounding box
            end_bbox: [4] ending bounding box
            num_frames: Number of frames to generate
            trajectory_type: 'linear', 'smooth', or 'accelerate'
            
        Returns:
            trajectory: [num_frames, 4] interpolated bounding boxes
        """
        if trajectory_type == 'linear':
            # Simple linear interpolation
            alpha = np.linspace(0, 1, num_frames)[:, np.newaxis]
            trajectory = start_bbox + alpha * (end_bbox - start_bbox)
            
        elif trajectory_type == 'smooth':
            # Smooth interpolation using quadratic (cubic needs 4+ points)
            t_in = np.array([0, num_frames - 1])
            t_out = np.linspace(0, num_frames - 1, num_frames)
            bboxes_in = np.stack([start_bbox, end_bbox])
            
            trajectory = []
            for dim in range(4):
                # Use quadratic for 2 points (or linear as fallback)
                try:
                    f = interp1d(t_in, bboxes_in[:, dim], kind='quadratic')
                except:
                    f = interp1d(t_in, bboxes_in[:, dim], kind='linear')
                trajectory.append(f(t_out))
            trajectory = np.stack(trajectory, axis=1)
            
        elif trajectory_type == 'accelerate':
            # Accelerating motion (ease-in-out)
            alpha = np.linspace(0, 1, num_frames)
            alpha = 3 * alpha**2 - 2 * alpha**3  # Smoothstep function
            alpha = alpha[:, np.newaxis]
            trajectory = start_bbox + alpha * (end_bbox - start_bbox)
        
        return trajectory
    
    def smooth_trajectory(
        self,
        trajectory: np.ndarray
    ) -> np.ndarray:
        """
        Apply smoothing to trajectory using moving average
        
        Args:
            trajectory: [T, 4] raw trajectory
            
        Returns:
            smoothed: [T, 4] smoothed trajectory
        """
        window = self.smoothing_window
        kernel = np.ones(window) / window
        
        smoothed = np.zeros_like(trajectory)
        for dim in range(4):
            smoothed[:, dim] = np.convolve(
                trajectory[:, dim], 
                kernel, 
                mode='same'
            )
        
        return smoothed
    
    def generate_trajectories(
        self,
        objects: List[Dict],
        num_frames: int,
        resolve_collisions: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete trajectories for all objects
        
        Args:
            objects: List of object specifications:
                {
                    'name': str,
                    'start_bbox': [x1, y1, x2, y2],
                    'end_bbox': [x1, y1, x2, y2],
                    'trajectory_type': str,
                    'priority': int
                }
            num_frames: Number of frames to generate
            resolve_collisions: Whether to resolve collisions
            
        Returns:
            trajectories: Dict mapping object names to [T, 4] trajectories
        """
        trajectories = {}
        
        # Generate initial trajectories
        for obj in objects:
            traj = self.interpolate_trajectory(
                np.array(obj['start_bbox']),
                np.array(obj['end_bbox']),
                num_frames,
                obj.get('trajectory_type', 'smooth')
            )
            
            # Apply smoothing
            traj = self.smooth_trajectory(traj)
            
            trajectories[obj['name']] = traj
        
        # Resolve collisions frame by frame
        if resolve_collisions:
            for frame_idx in range(num_frames):
                frame_bboxes = [
                    torch.tensor(traj[frame_idx]) 
                    for traj in trajectories.values()
                ]
                
                adjusted_bboxes = self.collision_detector.resolve(frame_bboxes)
                
                # Update trajectories with adjusted positions
                for obj_idx, obj_name in enumerate(trajectories.keys()):
                    trajectories[obj_name][frame_idx] = adjusted_bboxes[obj_idx].numpy()
        
        return trajectories
    
    def compute_trajectory_metrics(
        self,
        trajectories: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute trajectory quality metrics
        
        Args:
            trajectories: Dict of object trajectories
            
        Returns:
            metrics: Dict of metric values
        """
        metrics = {}
        
        # Smoothness: average acceleration magnitude
        smoothness_scores = []
        for name, traj in trajectories.items():
            velocity = np.diff(traj, axis=0)
            acceleration = np.diff(velocity, axis=0)
            smoothness = np.mean(np.linalg.norm(acceleration, axis=1))
            smoothness_scores.append(smoothness)
        metrics['smoothness'] = float(np.mean(smoothness_scores))
        
        # Collision score: percentage of frames with collisions
        num_frames = len(next(iter(trajectories.values())))
        collision_count = 0
        for frame_idx in range(num_frames):
            frame_bboxes = [
                torch.tensor(traj[frame_idx]) 
                for traj in trajectories.values()
            ]
            collisions = self.collision_detector.detect_collisions(frame_bboxes)
            if collisions:
                collision_count += 1
        metrics['collision_ratio'] = collision_count / num_frames
        
        return metrics