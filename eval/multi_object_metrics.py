"""
Comprehensive Multi-Object Video Generation Metrics
Novel metrics for evaluating multi-object trajectory control
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
import cv2
from torchvision.ops import box_iou


class MultiObjectMetrics:
    """
    Comprehensive metrics for multi-object video generation evaluation
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def compute_miou(
        self, 
        pred_bboxes: List[torch.Tensor], 
        gt_bboxes: List[torch.Tensor]
    ) -> float:
        """
        Mean Intersection over Union (standard Peekaboo metric)
        
        Args:
            pred_bboxes: List of [4] predicted bboxes per frame
            gt_bboxes: List of [4] ground truth bboxes per frame
            
        Returns:
            miou: Mean IoU across all frames
        """
        ious = []
        for pred, gt in zip(pred_bboxes, gt_bboxes):
            iou = self._compute_single_iou(pred, gt)
            ious.append(iou)
        
        return float(np.mean(ious))
    
    def _compute_single_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            intersection = 0.0
        else:
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        return float(iou)
    
    def compute_multi_object_iou(
        self,
        pred_objects: List[List[torch.Tensor]],  # [num_objects][num_frames][4]
        gt_objects: List[List[torch.Tensor]]
    ) -> float:
        """
        Multi-Object IoU (MOI) - Novel metric
        Average IoU across all objects simultaneously
        
        Args:
            pred_objects: List of predicted trajectories for each object
            gt_objects: List of ground truth trajectories for each object
            
        Returns:
            moi: Multi-Object IoU score
        """
        num_frames = len(pred_objects[0])
        frame_ious = []
        
        for frame_idx in range(num_frames):
            # Get all objects at this frame
            pred_frame = [obj[frame_idx] for obj in pred_objects]
            gt_frame = [obj[frame_idx] for obj in gt_objects]
            
            # Compute optimal matching between predicted and GT objects
            iou = self._match_and_compute_iou(pred_frame, gt_frame)
            frame_ious.append(iou)
        
        return float(np.mean(frame_ious))
    
    def _match_and_compute_iou(
        self, 
        pred_bboxes: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor]
    ) -> float:
        """Match predicted and GT boxes using Hungarian algorithm"""
        n_pred = len(pred_bboxes)
        n_gt = len(gt_bboxes)
        
        if n_pred == 0 or n_gt == 0:
            return 0.0
        
        # Compute cost matrix (negative IoU for maximization)
        cost_matrix = np.zeros((n_pred, n_gt))
        for i, pred in enumerate(pred_bboxes):
            for j, gt in enumerate(gt_bboxes):
                cost_matrix[i, j] = -self._compute_single_iou(pred, gt)
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute average IoU of matched pairs
        matched_ious = [-cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
        return float(np.mean(matched_ious))
    
    def compute_object_interaction_score(
        self,
        pred_objects: List[List[torch.Tensor]],
        gt_objects: List[List[torch.Tensor]]
    ) -> float:
        """
        Object Interaction Score (OIS) - Novel metric
        Measures how well object spatial relationships are preserved
        
        Args:
            pred_objects: Predicted object trajectories
            gt_objects: Ground truth object trajectories
            
        Returns:
            ois: Object interaction score (0-1, higher is better)
        """
        num_frames = len(pred_objects[0])
        num_objects = len(pred_objects)
        
        if num_objects < 2:
            return 1.0  # No interactions to measure
        
        interaction_scores = []
        
        for frame_idx in range(num_frames):
            pred_frame = [obj[frame_idx] for obj in pred_objects]
            gt_frame = [obj[frame_idx] for obj in gt_objects]
            
            # Compute pairwise distances
            pred_distances = self._compute_pairwise_distances(pred_frame)
            gt_distances = self._compute_pairwise_distances(gt_frame)
            
            # Compare distance matrices
            distance_diff = np.abs(pred_distances - gt_distances)
            score = np.exp(-distance_diff.mean())
            interaction_scores.append(score)
        
        return float(np.mean(interaction_scores))
    
    def _compute_pairwise_distances(
        self, 
        bboxes: List[torch.Tensor]
    ) -> np.ndarray:
        """Compute pairwise center distances between objects"""
        n = len(bboxes)
        distances = np.zeros((n, n))
        
        centers = []
        for bbox in bboxes:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        centers = np.array(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def compute_trajectory_consistency_score(
        self,
        pred_objects: List[List[torch.Tensor]]
    ) -> float:
        """
        Trajectory Consistency Score (TCS) - Novel metric
        Measures smoothness of trajectories
        
        Args:
            pred_objects: Predicted object trajectories
            
        Returns:
            tcs: Trajectory consistency score (0-1, higher is better)
        """
        consistency_scores = []
        
        for obj_trajectory in pred_objects:
            # Convert to numpy array
            trajectory = torch.stack(obj_trajectory).numpy()  # [T, 4]
            
            # Compute velocities and accelerations
            velocity = np.diff(trajectory, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # Smoothness: low acceleration magnitude
            acc_magnitude = np.linalg.norm(acceleration, axis=1)
            smoothness = np.exp(-acc_magnitude.mean())
            
            # Direction consistency: how much direction changes
            if len(velocity) > 1:
                velocity_normalized = velocity / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-6)
                direction_changes = np.linalg.norm(np.diff(velocity_normalized, axis=0), axis=1)
                direction_consistency = np.exp(-direction_changes.mean())
            else:
                direction_consistency = 1.0
            
            # Combine metrics
            score = 0.7 * smoothness + 0.3 * direction_consistency
            consistency_scores.append(score)
        
        return float(np.mean(consistency_scores))
    
    def compute_collision_avoidance_score(
        self,
        pred_objects: List[List[torch.Tensor]]
    ) -> float:
        """
        Collision Avoidance Score (CAS) - Novel metric
        Measures physical plausibility (no overlaps)
        
        Args:
            pred_objects: Predicted object trajectories
            
        Returns:
            cas: Collision avoidance score (0-1, higher is better)
        """
        num_frames = len(pred_objects[0])
        num_objects = len(pred_objects)
        
        if num_objects < 2:
            return 1.0  # No collisions possible
        
        collision_free_frames = 0
        
        for frame_idx in range(num_frames):
            frame_bboxes = [obj[frame_idx] for obj in pred_objects]
            has_collision = False
            
            # Check all pairs
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    iou = self._compute_single_iou(frame_bboxes[i], frame_bboxes[j])
                    if iou > 0.1:  # Significant overlap
                        has_collision = True
                        break
                if has_collision:
                    break
            
            if not has_collision:
                collision_free_frames += 1
        
        cas = collision_free_frames / num_frames
        return float(cas)
    
    def compute_coverage(
        self,
        pred_objects: List[List[torch.Tensor]],
        detection_threshold: float = 0.3
    ) -> float:
        """
        Coverage metric (from Peekaboo)
        Fraction of frames where objects are successfully detected
        
        Args:
            pred_objects: Predicted object trajectories
            detection_threshold: Minimum IoU to consider detected
            
        Returns:
            coverage: Coverage score (0-1)
        """
        num_frames = len(pred_objects[0])
        num_objects = len(pred_objects)
        
        detected_frames = 0
        
        for frame_idx in range(num_frames):
            frame_detected = True
            for obj in pred_objects:
                bbox = obj[frame_idx]
                # Check if bbox is valid (not degenerate)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < 0.01 or height < 0.01:
                    frame_detected = False
                    break
            
            if frame_detected:
                detected_frames += 1
        
        coverage = detected_frames / num_frames
        return float(coverage)
    
    def compute_ap50(
        self,
        pred_bboxes: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor]
    ) -> float:
        """
        Average Precision @ 50% IoU (from Peekaboo)
        
        Args:
            pred_bboxes: Predicted bounding boxes
            gt_bboxes: Ground truth bounding boxes
            
        Returns:
            ap50: Average Precision at 50% IoU threshold
        """
        tp = 0
        total = len(gt_bboxes)
        
        for pred, gt in zip(pred_bboxes, gt_bboxes):
            iou = self._compute_single_iou(pred, gt)
            if iou >= 0.5:
                tp += 1
        
        ap50 = tp / total if total > 0 else 0.0
        return float(ap50)
    
    def compute_centroid_distance(
        self,
        pred_bboxes: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor]
    ) -> float:
        """
        Centroid Distance (from Peekaboo)
        Normalized distance between predicted and GT centroids
        
        Args:
            pred_bboxes: Predicted bounding boxes
            gt_bboxes: Ground truth bounding boxes
            
        Returns:
            cd: Average centroid distance (normalized)
        """
        distances = []
        
        for pred, gt in zip(pred_bboxes, gt_bboxes):
            pred_center = torch.tensor([
                (pred[0] + pred[2]) / 2,
                (pred[1] + pred[3]) / 2
            ])
            gt_center = torch.tensor([
                (gt[0] + gt[2]) / 2,
                (gt[1] + gt[3]) / 2
            ])
            
            dist = torch.norm(pred_center - gt_center)
            distances.append(dist.item())
        
        cd = np.mean(distances)
        return float(cd)
    
    def compute_species_specific_accuracy(
        self,
        pred_objects: List[Dict],  # Each dict has 'species', 'trajectory'
        gt_objects: List[Dict],
        motion_priors: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Species-Specific Accuracy (SSA) - Novel metric
        Measures how well motion matches expected species behavior
        
        Args:
            pred_objects: Predicted objects with species labels
            gt_objects: Ground truth objects with species labels
            motion_priors: Dictionary of motion statistics per species
            
        Returns:
            ssa_scores: Dictionary mapping species to accuracy scores
        """
        ssa_scores = {}
        
        for pred_obj, gt_obj in zip(pred_objects, gt_objects):
            species = pred_obj['species']
            pred_traj = pred_obj['trajectory']  # [T, 4] numpy array
            
            if species not in motion_priors:
                continue
            
            prior = motion_priors[species]
            
            # Compute velocity characteristics
            velocity = np.diff(pred_traj, axis=0)
            mean_vel = np.mean(velocity, axis=0)
            
            # Compare with prior
            expected_vel = prior['mean_velocity']
            vel_error = np.linalg.norm(mean_vel - expected_vel)
            vel_score = np.exp(-vel_error)
            
            # Compute size characteristics
            widths = pred_traj[:, 2] - pred_traj[:, 0]
            heights = pred_traj[:, 3] - pred_traj[:, 1]
            mean_width = np.mean(widths)
            mean_height = np.mean(heights)
            
            # Compare with prior
            width_error = abs(mean_width - prior['mean_width']) / (prior['std_width'] + 1e-6)
            height_error = abs(mean_height - prior['mean_height']) / (prior['std_height'] + 1e-6)
            size_score = np.exp(-(width_error + height_error) / 2)
            
            # Combine scores
            score = 0.6 * vel_score + 0.4 * size_score
            
            if species not in ssa_scores:
                ssa_scores[species] = []
            ssa_scores[species].append(score)
        
        # Average per species
        ssa_scores = {k: float(np.mean(v)) for k, v in ssa_scores.items()}
        
        return ssa_scores
    
    def compute_all_metrics(
        self,
        pred_objects: List[List[torch.Tensor]],
        gt_objects: List[List[torch.Tensor]],
        species_info: Optional[List[str]] = None,
        motion_priors: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics at once
        
        Args:
            pred_objects: Predicted object trajectories
            gt_objects: Ground truth object trajectories
            species_info: Optional species labels for SSA
            motion_priors: Optional motion priors for SSA
            
        Returns:
            metrics: Dictionary of all computed metrics
        """
        metrics = {}
        
        # Standard Peekaboo metrics
        if len(pred_objects) == 1 and len(gt_objects) == 1:
            metrics['mIoU'] = self.compute_miou(pred_objects[0], gt_objects[0])
            metrics['AP50'] = self.compute_ap50(pred_objects[0], gt_objects[0])
            metrics['CD'] = self.compute_centroid_distance(pred_objects[0], gt_objects[0])
        
        metrics['Coverage'] = self.compute_coverage(pred_objects)
        
        # Novel multi-object metrics
        if len(pred_objects) > 1:
            metrics['MOI'] = self.compute_multi_object_iou(pred_objects, gt_objects)
            metrics['OIS'] = self.compute_object_interaction_score(pred_objects, gt_objects)
            metrics['CAS'] = self.compute_collision_avoidance_score(pred_objects)
        
        metrics['TCS'] = self.compute_trajectory_consistency_score(pred_objects)
        
        # Species-specific accuracy (if provided)
        if species_info is not None and motion_priors is not None:
            pred_with_species = [
                {'species': species, 'trajectory': torch.stack(traj).numpy()}
                for species, traj in zip(species_info, pred_objects)
            ]
            gt_with_species = [
                {'species': species, 'trajectory': torch.stack(traj).numpy()}
                for species, traj in zip(species_info, gt_objects)
            ]
            ssa = self.compute_species_specific_accuracy(
                pred_with_species, 
                gt_with_species, 
                motion_priors
            )
            metrics['SSA'] = ssa
        
        return metrics


class VideoQualityMetrics:
    """Additional video quality metrics (FVD, etc.)"""
    
    def __init__(self):
        pass
    
    def compute_fvd(
        self,
        pred_videos: torch.Tensor,
        gt_videos: torch.Tensor
    ) -> float:
        """
        Fréchet Video Distance (requires I3D features)
        Simplified version - full implementation needs pretrained I3D
        
        Args:
            pred_videos: [B, T, C, H, W] predicted videos
            gt_videos: [B, T, C, H, W] ground truth videos
            
        Returns:
            fvd: FVD score
        """
        # This is a placeholder - full FVD requires I3D model
        # For now, compute simple feature distance
        
        pred_mean = pred_videos.mean(dim=[0, 1, 3, 4])
        gt_mean = gt_videos.mean(dim=[0, 1, 3, 4])
        
        fvd = torch.norm(pred_mean - gt_mean).item()
        
        return float(fvd)