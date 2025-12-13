import os
import json
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

class AnimalKingdomSubsetPreparer:
    def __init__(self, source_dir, output_dir, target_hours=3.5):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_seconds = target_hours * 3600
        self.fps = 30
        
    def scan_videos(self):
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(list(self.source_dir.rglob(ext)))
        return video_files
    
    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    
    def extract_bboxes_from_annotations(self, video_path, annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        video_name = video_path.stem
        if video_name not in annotations:
            return None
            
        video_data = annotations[video_name]
        bboxes = []
        
        for frame_idx, frame_data in enumerate(video_data.get('frames', [])):
            frame_bboxes = []
            for obj in frame_data.get('objects', []):
                bbox = obj.get('bbox', [])
                species = obj.get('species', 'unknown')
                if len(bbox) == 4:
                    frame_bboxes.append({
                        'bbox': bbox,
                        'species': species,
                        'frame': frame_idx
                    })
            bboxes.append(frame_bboxes)
        
        return bboxes
    
    def select_subset(self):
        all_videos = self.scan_videos()
        random.shuffle(all_videos)
        
        selected_videos = []
        total_duration = 0
        
        for video_path in all_videos:
            if total_duration >= self.target_seconds:
                break
                
            duration = self.get_video_duration(video_path)
            if duration > 0:
                selected_videos.append({
                    'path': video_path,
                    'duration': duration
                })
                total_duration += duration
        
        return selected_videos, total_duration
    
    def prepare_dataset(self, annotation_file=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        selected_videos, total_duration = self.select_subset()
        
        print(f"Selected {len(selected_videos)} videos")
        print(f"Total duration: {total_duration/3600:.2f} hours")
        
        dataset_info = []
        
        for idx, video_info in enumerate(selected_videos):
            video_path = video_info['path']
            dest_path = self.output_dir / f"video_{idx:04d}.mp4"
            
            shutil.copy(video_path, dest_path)
            
            bboxes = None
            if annotation_file:
                bboxes = self.extract_bboxes_from_annotations(video_path, annotation_file)
            
            dataset_info.append({
                'video_id': f"video_{idx:04d}",
                'original_path': str(video_path),
                'duration': video_info['duration'],
                'bboxes': bboxes
            })
        
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info

if __name__ == "__main__":
    preparer = AnimalKingdomSubsetPreparer(
        source_dir="/path/to/animal_kingdom",
        output_dir="./data/animal_kingdom_subset",
        target_hours=3.5
    )
    
    dataset_info = preparer.prepare_dataset(
        annotation_file="/path/to/animal_kingdom/annotations.json"
    )
    
    print(f"Dataset prepared with {len(dataset_info)} videos")