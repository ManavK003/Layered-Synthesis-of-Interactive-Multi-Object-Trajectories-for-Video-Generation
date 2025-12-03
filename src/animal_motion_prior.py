"""
Animal-Specific Motion Priors from Animal Kingdom Dataset
Learns species-specific motion patterns
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json
import pickle
from collections import defaultdict


class AnimalMotionEncoder(nn.Module):
    """Encode animal motion patterns into embeddings"""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Encoder for motion sequences
        self.motion_encoder = nn.Sequential(
            nn.Linear(4, 64),  # bbox coordinates
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # Temporal aggregation
        self.temporal_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode motion sequence into embedding
        
        Args:
            motion_sequence: [B, T, 4] bounding box sequence
            
        Returns:
            embedding: [B, embedding_dim] motion embedding
        """
        B, T, _ = motion_sequence.shape
        
        # Encode each frame
        frame_embeddings = []
        for t in range(T):
            frame_emb = self.motion_encoder(motion_sequence[:, t, :])
            frame_embeddings.append(frame_emb)
        frame_embeddings = torch.stack(frame_embeddings, dim=1)  # [B, T, D]
        
        # Temporal aggregation
        _, (hidden, _) = self.temporal_lstm(frame_embeddings)
        embedding = hidden[-1]  # [B, D]
        
        return embedding


class AnimalMotionPrior:
    """
    Extract and store motion priors from Animal Kingdom dataset
    """
    
    def __init__(
        self, 
        embedding_dim: int = 256,
        cache_path: str = './data/animal_motion_cache.pkl'
    ):
        self.embedding_dim = embedding_dim
        self.cache_path = cache_path
        self.encoder = AnimalMotionEncoder(embedding_dim)
        
        # Storage for motion statistics
        self.species_motions = defaultdict(list)
        self.species_embeddings = {}
        self.motion_statistics = {}
        
    def extract_from_animal_kingdom(
        self,
        animal_kingdom_data_path: str,
        split: str = 'train'
    ):
        """
        Extract motion patterns from Animal Kingdom dataset
        
        Args:
            animal_kingdom_data_path: Path to Animal Kingdom dataset
            split: 'train', 'val', or 'test'
        """
        print(f"Extracting motion patterns from Animal Kingdom ({split} split)...")
        
        # Load Animal Kingdom annotations
        annotation_path = f"{animal_kingdom_data_path}/action_recognition/annotation/{split}.json"
        
        try:
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {annotation_path}")
            print("Creating dummy data for testing...")
            annotations = self._create_dummy_annotations()
        
        # Process each video
        for video_id, video_data in annotations.items():
            species = video_data.get('species', 'unknown')
            bboxes = video_data.get('bboxes', [])
            
            if len(bboxes) > 0:
                self.species_motions[species].append(np.array(bboxes))
        
        # Compute statistics for each species
        print("Computing motion statistics...")
        for species, motion_list in self.species_motions.items():
            if len(motion_list) > 0:
                self.motion_statistics[species] = self._compute_motion_stats(
                    motion_list
                )
        
        print(f"Extracted motion patterns for {len(self.species_motions)} species")
        
    def _create_dummy_annotations(self) -> Dict:
        """Create dummy annotations for testing"""
        dummy_annotations = {}
        
        species_list = ['lion', 'zebra', 'elephant', 'giraffe', 'cheetah']
        
        for i in range(50):
            species = np.random.choice(species_list)
            num_frames = np.random.randint(20, 50)
            
            # Generate random trajectory
            start_x = np.random.uniform(0.1, 0.4)
            start_y = np.random.uniform(0.1, 0.4)
            end_x = np.random.uniform(0.6, 0.9)
            end_y = np.random.uniform(0.6, 0.9)
            
            bboxes = []
            for t in range(num_frames):
                alpha = t / num_frames
                x1 = start_x + alpha * (end_x - start_x)
                y1 = start_y + alpha * (end_y - start_y)
                w = np.random.uniform(0.1, 0.2)
                h = np.random.uniform(0.1, 0.2)
                
                bboxes.append([x1, y1, x1 + w, y1 + h])
            
            dummy_annotations[f'video_{i}'] = {
                'species': species,
                'bboxes': bboxes
            }
        
        return dummy_annotations
    
    def _compute_motion_stats(
        self, 
        motion_list: List[np.ndarray]
    ) -> Dict:
        """
        Compute statistical summaries of motion patterns
        
        Args:
            motion_list: List of [T, 4] motion sequences
            
        Returns:
            stats: Dictionary of motion statistics
        """
        stats = {}
        
        # Compute velocity statistics
        velocities = []
        for motion in motion_list:
            velocity = np.diff(motion, axis=0)
            velocities.append(velocity)
        velocities = np.concatenate(velocities, axis=0)
        
        stats['mean_velocity'] = np.mean(velocities, axis=0)
        stats['std_velocity'] = np.std(velocities, axis=0)
        
        # Compute typical bbox sizes
        all_bboxes = np.concatenate(motion_list, axis=0)
        widths = all_bboxes[:, 2] - all_bboxes[:, 0]
        heights = all_bboxes[:, 3] - all_bboxes[:, 1]
        
        stats['mean_width'] = np.mean(widths)
        stats['std_width'] = np.std(widths)
        stats['mean_height'] = np.mean(heights)
        stats['std_height'] = np.std(heights)
        
        # Compute motion range
        stats['motion_range_x'] = (
            np.percentile(all_bboxes[:, 0], 95) - 
            np.percentile(all_bboxes[:, 0], 5)
        )
        stats['motion_range_y'] = (
            np.percentile(all_bboxes[:, 1], 95) - 
            np.percentile(all_bboxes[:, 1], 5)
        )
        
        return stats
    
    def get_motion_prior(self, species: str) -> Dict:
        """
        Get motion prior for specific species
        
        Args:
            species: Species name
            
        Returns:
            prior: Motion statistics for the species
        """
        if species in self.motion_statistics:
            return self.motion_statistics[species]
        else:
            # Return generic prior
            print(f"Warning: No motion prior for species '{species}', using generic")
            return self._get_generic_prior()
    
    def _get_generic_prior(self) -> Dict:
        """Return generic motion prior averaged across all species"""
        if len(self.motion_statistics) == 0:
            # Default values
            return {
                'mean_velocity': np.array([0.01, 0.01, 0.01, 0.01]),
                'std_velocity': np.array([0.02, 0.02, 0.02, 0.02]),
                'mean_width': 0.15,
                'std_width': 0.05,
                'mean_height': 0.15,
                'std_height': 0.05,
                'motion_range_x': 0.5,
                'motion_range_y': 0.5
            }
        
        # Average across all species
        all_stats = list(self.motion_statistics.values())
        generic = {}
        
        for key in all_stats[0].keys():
            values = [stats[key] for stats in all_stats]
            generic[key] = np.mean(values, axis=0)
        
        return generic
    
    def adjust_trajectory_with_prior(
        self,
        trajectory: np.ndarray,
        species: str,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Adjust trajectory using species-specific motion prior
        
        Args:
            trajectory: [T, 4] original trajectory
            species: Species name
            strength: How much to adjust (0=no change, 1=full adjustment)
            
        Returns:
            adjusted_trajectory: [T, 4] adjusted trajectory
        """
        prior = self.get_motion_prior(species)
        adjusted = trajectory.copy()
        
        # Adjust bbox sizes to match typical sizes
        for t in range(len(trajectory)):
            width = trajectory[t, 2] - trajectory[t, 0]
            height = trajectory[t, 3] - trajectory[t, 1]
            center_x = (trajectory[t, 0] + trajectory[t, 2]) / 2
            center_y = (trajectory[t, 1] + trajectory[t, 3]) / 2
            
            # Blend with prior
            target_width = prior['mean_width']
            target_height = prior['mean_height']
            
            new_width = width * (1 - strength) + target_width * strength
            new_height = height * (1 - strength) + target_height * strength
            
            adjusted[t, 0] = center_x - new_width / 2
            adjusted[t, 1] = center_y - new_height / 2
            adjusted[t, 2] = center_x + new_width / 2
            adjusted[t, 3] = center_y + new_height / 2
        
        # Smooth velocities based on prior
        velocity = np.diff(adjusted, axis=0)
        mean_vel = prior['mean_velocity']
        
        for t in range(len(velocity)):
            velocity[t] = (
                velocity[t] * (1 - strength * 0.3) + 
                mean_vel * (strength * 0.3)
            )
        
        # Integrate velocities back
        for t in range(1, len(adjusted)):
            adjusted[t] = adjusted[t-1] + velocity[t-1]
        
        return adjusted
    
    def save_cache(self):
        """Save motion priors to cache file"""
        cache_data = {
            'motion_statistics': self.motion_statistics,
            'species_motions': {k: v for k, v in self.species_motions.items()}
        }
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Saved motion prior cache to {self.cache_path}")
    
    def load_cache(self):
        """Load motion priors from cache file"""
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.motion_statistics = cache_data['motion_statistics']
            self.species_motions = defaultdict(list, cache_data['species_motions'])
            
            print(f"Loaded motion prior cache from {self.cache_path}")
            return True
        except FileNotFoundError:
            print(f"No cache found at {self.cache_path}")
            return False