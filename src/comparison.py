import torch
import cv2
import numpy as np
from pathlib import Path
import json
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class VideoEvaluator:
    def __init__(self, model_name="google/owlvit-large-patch14", device="cuda"):
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        
    def detect_objects(self, frame, text_queries, threshold=0.1):
        image = Image.fromarray(frame)
        
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            detections.append({
                'bbox': box,
                'score': score.item(),
                'label': label.item()
            })
        
        return detections
    
    def compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def compute_centroid_distance(self, box1, box2):
        c1_x = (box1[0] + box1[2]) / 2
        c1_y = (box1[1] + box1[3]) / 2
        c2_x = (box2[0] + box2[2]) / 2
        c2_y = (box2[1] + box2[3]) / 2
        
        distance = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        
        return distance
    
    def evaluate_video(self, video_path, ground_truth_bboxes, text_queries):
        cap = cv2.VideoCapture(str(video_path))
        
        frame_detections = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detections = self.detect_objects(frame_rgb, text_queries)
            frame_detections.append(detections)
            
            frame_idx += 1
        
        cap.release()
        
        ious = []
        centroid_distances = []
        detected_frames = 0
        
        for frame_idx, (detections, gt_bboxes) in enumerate(zip(frame_detections, ground_truth_bboxes)):
            if not detections or not gt_bboxes:
                continue
            
            detected_frames += 1
            
            for gt_bbox in gt_bboxes:
                best_iou = 0
                best_distance = float('inf')
                
                for det in detections:
                    iou = self.compute_iou(det['bbox'], gt_bbox)
                    distance = self.compute_centroid_distance(det['bbox'], gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_distance = distance
                
                ious.append(best_iou)
                centroid_distances.append(best_distance)
        
        coverage = detected_frames / len(frame_detections) if frame_detections else 0
        mean_iou = np.mean(ious) if ious else 0
        mean_cd = np.mean(centroid_distances) if centroid_distances else 0
        ap50 = sum(1 for iou in ious if iou > 0.5) / len(ious) if ious else 0
        
        return {
            'mIOU': mean_iou,
            'Coverage': coverage,
            'CD': mean_cd,
            'AP50': ap50
        }

def evaluate_and_compare(zbuffer_dir, peekaboo_dir, dataset_info_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = VideoEvaluator()
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    zbuffer_results = []
    peekaboo_results = []
    
    zbuffer_videos = list(Path(zbuffer_dir).glob("*.mp4"))
    peekaboo_videos = list(Path(peekaboo_dir).glob("*.mp4"))
    
    print("Evaluating Z-buffer videos...")
    for video_path in tqdm(zbuffer_videos):
        video_id = video_path.stem.split('_')[-1]
        
        gt_info = next((item for item in dataset_info if item['video_id'] == video_id), None)
        if not gt_info or not gt_info.get('bboxes'):
            continue
        
        gt_bboxes = []
        for frame_data in gt_info['bboxes']:
            if frame_data:
                frame_bboxes = [obj['bbox'] for obj in frame_data]
                gt_bboxes.append(frame_bboxes)
            else:
                gt_bboxes.append([])
        
        text_queries = ["animal", "wildlife", "creature"]
        
        metrics = evaluator.evaluate_video(video_path, gt_bboxes, text_queries)
        metrics['video_id'] = video_id
        zbuffer_results.append(metrics)
    
    print("Evaluating Peekaboo videos...")
    for video_path in tqdm(peekaboo_videos):
        video_id = video_path.stem.split('_')[-1]
        
        gt_info = next((item for item in dataset_info if item['video_id'] == video_id), None)
        if not gt_info or not gt_info.get('bboxes'):
            continue
        
        gt_bboxes = []
        for frame_data in gt_info['bboxes']:
            if frame_data:
                frame_bboxes = [obj['bbox'] for obj in frame_data]
                gt_bboxes.append(frame_bboxes)
            else:
                gt_bboxes.append([])
        
        text_queries = ["animal", "wildlife", "creature"]
        
        metrics = evaluator.evaluate_video(video_path, gt_bboxes, text_queries)
        metrics['video_id'] = video_id
        peekaboo_results.append(metrics)
    
    zbuffer_avg = {
        'mIOU': np.mean([r['mIOU'] for r in zbuffer_results]),
        'Coverage': np.mean([r['Coverage'] for r in zbuffer_results]),
        'CD': np.mean([r['CD'] for r in zbuffer_results]),
        'AP50': np.mean([r['AP50'] for r in zbuffer_results])
    }
    
    peekaboo_avg = {
        'mIOU': np.mean([r['mIOU'] for r in peekaboo_results]),
        'Coverage': np.mean([r['Coverage'] for r in peekaboo_results]),
        'CD': np.mean([r['CD'] for r in peekaboo_results]),
        'AP50': np.mean([r['AP50'] for r in peekaboo_results])
    }
    
    comparison = {
        'zbuffer': {
            'results': zbuffer_results,
            'average': zbuffer_avg
        },
        'peekaboo': {
            'results': peekaboo_results,
            'average': peekaboo_avg
        }
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    metrics = ['mIOU', 'Coverage', 'CD', 'AP50']
    zbuffer_vals = [zbuffer_avg[m] for m in metrics]
    peekaboo_vals = [peekaboo_avg[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, zbuffer_vals, width, label='Z-buffer (Ours)', color='#4facfe')
    bars2 = ax.bar(x + width/2, peekaboo_vals, width, label='Peekaboo', color='#f093fb')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Z-buffer vs Peekaboo Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_chart.png', dpi=300)
    plt.close()
    
    print("\n=== Comparison Results ===")
    print(f"\nZ-buffer (Ours):")
    for metric, value in zbuffer_avg.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nPeekaboo:")
    for metric, value in peekaboo_avg.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nImprovement:")
    for metric in metrics:
        improvement = ((zbuffer_avg[metric] - peekaboo_avg[metric]) / peekaboo_avg[metric]) * 100
        print(f"  {metric}: {improvement:+.2f}%")
    
    return comparison

if __name__ == "__main__":
    comparison = evaluate_and_compare(
        zbuffer_dir="./outputs/zbuffer_training",
        peekaboo_dir="./outputs/peekaboo_training",
        dataset_info_path="./data/animal_kingdom_subset/dataset_info.json",
        output_dir="./outputs/evaluation"
    )