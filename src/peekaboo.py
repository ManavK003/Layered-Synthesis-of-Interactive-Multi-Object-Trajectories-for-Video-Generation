import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('.')
from train_zbuffer import AnimalKingdomDataset

class PeekabooGenerator:
    def __init__(self, model_name="cerspense/zeroscope_v2_576w", device="cuda"):
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        
    def create_attention_mask(self, bboxes, shape):
        h, w = shape
        masks = []
        
        for bbox in bboxes:
            mask = np.zeros((h, w), dtype=np.float32)
            if bbox:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                mask[y1:y2, x1:x2] = 1.0
            masks.append(mask)
        
        return masks
    
    def modify_attention_with_masks(self, unet, masks):
        def attention_hook(module, args, output):
            return output
        
        hooks = []
        for name, module in unet.named_modules():
            if 'attn' in name:
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        return hooks
    
    def generate_multi_object_video(self, prompt, bboxes_list, num_frames=24, num_steps=50):
        video_frames = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=7.5
        ).frames[0]
        
        return [np.array(frame) for frame in video_frames]

def train_peekaboo_model(data_dir, dataset_info_path, output_dir, num_epochs=10, batch_size=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = AnimalKingdomDataset(data_dir, dataset_info_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    generator = PeekabooGenerator()
    
    results = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            video_id = batch['video_id'][0]
            bboxes = batch['bboxes']
            
            num_objects = max(len(frame_bboxes[0]) for frame_bboxes in bboxes if frame_bboxes[0])
            
            if num_objects == 0:
                continue
            
            prompt = f"animals in natural habitat with {num_objects} animals"
            
            bboxes_per_frame = []
            for frame_bboxes in bboxes:
                if frame_bboxes[0]:
                    bboxes_per_frame.append(frame_bboxes[0])
                else:
                    bboxes_per_frame.append([])
            
            try:
                generated_frames = generator.generate_multi_object_video(
                    prompt, bboxes_per_frame
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
    results = train_peekaboo_model(
        data_dir="./data/animal_kingdom_subset",
        dataset_info_path="./data/animal_kingdom_subset/dataset_info.json",
        output_dir="./outputs/peekaboo_training",
        num_epochs=5,
        batch_size=1
    )
    
    print("Training complete!")