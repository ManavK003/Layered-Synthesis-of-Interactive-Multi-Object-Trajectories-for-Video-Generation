import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

class AnimalKingdomDataset(Dataset):
    def __init__(self, data_dir, dataset_info_path, num_frames=24, resolution=(320, 576)):
        self.data_dir = Path(data_dir)
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        self.num_frames = num_frames
        self.resolution = resolution
        
    def __len__(self):
        return len(self.dataset_info)
    
    def extract_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.resolution)
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return np.array(frames[:num_frames])
    
    def normalize_bbox(self, bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        return [x1/w, y1/h, x2/w, y2/h]
    
    def __getitem__(self, idx):
        info = self.dataset_info[idx]
        video_path = self.data_dir / f"{info['video_id']}.mp4"
        
        frames = self.extract_frames(video_path, self.num_frames)
        
        bboxes = info.get('bboxes', [])
        if bboxes and len(bboxes) > 0:
            frame_bboxes = []
            for frame_data in bboxes[:self.num_frames]:
                if frame_data:
                    normalized = [self.normalize_bbox(obj['bbox'], frames[0].shape) 
                                for obj in frame_data]
                    frame_bboxes.append(normalized)
                else:
                    frame_bboxes.append([])
        else:
            frame_bboxes = [[] for _ in range(self.num_frames)]
        
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        return {
            'frames': frames_tensor,
            'bboxes': frame_bboxes,
            'video_id': info['video_id']
        }

class ZBufferVideoGenerator:
    def __init__(self, model_name="cerspense/zeroscope_v2_576w", device="cuda"):
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        
    def generate_object_trajectory(self, prompt, bboxes, num_frames=24, num_steps=50):
        blank_prompt = f"{prompt} on blank white background"
        
        video_frames = self.pipe(
            prompt=blank_prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=7.5
        ).frames[0]
        
        object_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame_np = np.array(frame)
            if frame_idx < len(bboxes) and bboxes[frame_idx]:
                bbox = bboxes[frame_idx]
                x1, y1, x2, y2 = bbox
                h, w = frame_np.shape[:2]
                x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                
                masked_frame = cv2.bitwise_and(frame_np, frame_np, mask=mask)
                object_frames.append(masked_frame)
            else:
                object_frames.append(frame_np)
        
        return object_frames
    
    def zbuffer_fusion(self, background_frames, object_frames_list, z_values):
        fused_frames = []
        
        for frame_idx in range(len(background_frames)):
            zbuffer = np.full(background_frames[frame_idx].shape[:2], np.inf)
            output_frame = background_frames[frame_idx].copy()
            
            for obj_idx, object_frames in enumerate(object_frames_list):
                if frame_idx >= len(object_frames):
                    continue
                    
                obj_frame = object_frames[frame_idx]
                z_val = z_values[obj_idx]
                
                mask = np.any(obj_frame > 0, axis=-1)
                
                update_mask = (zbuffer > z_val) & mask
                
                zbuffer[update_mask] = z_val
                output_frame[update_mask] = obj_frame[update_mask]
            
            fused_frames.append(output_frame)
        
        return fused_frames
    
    def generate_multi_object_video(self, prompts, bboxes_list, background_prompt="blank white background"):
        background_frames = self.generate_object_trajectory(
            background_prompt, [[]], num_frames=24
        )
        
        object_frames_list = []
        for prompt, bboxes in zip(prompts, bboxes_list):
            obj_frames = self.generate_object_trajectory(prompt, bboxes, num_frames=24)
            object_frames_list.append(obj_frames)
        
        z_values = list(range(len(object_frames_list)))
        
        fused_frames = self.zbuffer_fusion(background_frames, object_frames_list, z_values)
        
        return fused_frames

def train_zbuffer_model(data_dir, dataset_info_path, output_dir, num_epochs=10, batch_size=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = AnimalKingdomDataset(data_dir, dataset_info_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    generator = ZBufferVideoGenerator()
    
    results = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            video_id = batch['video_id'][0]
            bboxes = batch['bboxes']
            
            num_objects = max(len(frame_bboxes[0]) for frame_bboxes in bboxes)
            
            if num_objects == 0:
                continue
            
            prompts = [f"animal {i+1}" for i in range(num_objects)]
            
            bboxes_per_object = []
            for obj_idx in range(num_objects):
                obj_bboxes = []
                for frame_bboxes in bboxes:
                    if frame_bboxes[0] and obj_idx < len(frame_bboxes[0]):
                        obj_bboxes.append(frame_bboxes[0][obj_idx])
                    else:
                        obj_bboxes.append([0, 0, 0.1, 0.1])
                bboxes_per_object.append(obj_bboxes)
            
            try:
                generated_frames = generator.generate_multi_object_video(
                    prompts, bboxes_per_object
                )
                
                output_path = output_dir / f"epoch_{epoch}_batch_{batch_idx}_{video_id}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, 8, (576, 320))
                for frame in generated_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                
                epoch_metrics.append({
                    'video_id': video_id,
                    'num_objects': num_objects,
                    'output_path': str(output_path)
                })
                
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                continue
        
        results.append({
            'epoch': epoch + 1,
            'metrics': epoch_metrics
        })
        
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = train_zbuffer_model(
        data_dir="./data/animal_kingdom_subset",
        dataset_info_path="./data/animal_kingdom_subset/dataset_info.json",
        output_dir="./outputs/zbuffer_training",
        num_epochs=5,
        batch_size=1
    )
    
    print("Training complete!")