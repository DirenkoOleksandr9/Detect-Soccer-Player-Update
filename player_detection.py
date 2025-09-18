import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
from tqdm import tqdm
from typing import List, Dict

# Force headless mode to avoid OpenGL issues
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'

class SoccerPlayerDetector:
    def __init__(self, model_name: str = 'yolov8x.pt', conf_thresh: float = 0.3, min_area: int = 500):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf_thresh = conf_thresh
        self.min_area = min_area
        self.target_classes = [0, 32]  # Person and sports ball
        print(f"Detector initialized on {self.device} with model {model_name}")
        print(f"Confidence threshold: {conf_thresh}, Min area: {min_area}")

    def process_video(self, video_path: str, output_path: str) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        all_detections = []
        
        with tqdm(total=total_frames, desc="Stage 1: SOTA Detection") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: 
                    break
                
                results = self.model(frame, classes=self.target_classes, imgsz=1280, verbose=False)
                
                frame_detections = {'players': [], 'ball': None}
                
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.conf[0] >= self.conf_thresh:
                            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                            area = (x2 - x1) * (y2 - y1)
                            
                            if area < self.min_area:
                                continue
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2], 
                                'confidence': float(box.conf[0]),
                                'area': area,
                                'class': int(box.cls[0])
                            }
                            
                            if int(box.cls[0]) == 0:  # Person
                                frame_detections['players'].append(detection)
                            elif int(box.cls[0]) == 32:  # Sports Ball
                                if frame_detections['ball'] is None or detection['confidence'] > frame_detections['ball']['confidence']:
                                    frame_detections['ball'] = detection
                
                all_detections.append({
                    "frame_id": frame_idx, 
                    "detections": frame_detections,
                    "timestamp": frame_idx / fps if fps > 0 else 0
                })
                pbar.update(1)
        
        cap.release()
        
        with open(output_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        print(f"Detection complete. Saved to {output_path}")
        print(f"Total detections: {sum(len(frame['detections']['players']) for frame in all_detections)} players")
        
        return all_detections

if __name__ == "__main__":
    detector = SoccerPlayerDetector()
    detector.process_video("9.mp4", "detections.json")
