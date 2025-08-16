import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
from tqdm import tqdm
import os
from typing import List, Dict

# Force headless mode to avoid OpenGL issues
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'

class SoccerPlayerDetector:
    def __init__(self, model_name: str = 'yolov8n.pt', conf_thresh: float = 0.4, min_area: int = 1000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf_thresh = conf_thresh
        self.min_area = min_area
        self.target_classes = [0, 32] # 0 is person, 32 is sports ball

    def process_video(self, video_path: str, output_path: str) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_detections = []
        
        with tqdm(total=total_frames, desc="Stage 1: Detecting Players & Ball") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                results = self.model(frame, classes=self.target_classes, imgsz=960, verbose=False)
                
                frame_detections = {'players': [], 'ball': None}
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.conf[0] >= self.conf_thresh:
                            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                            if (x2 - x1) * (y2 - y1) < self.min_area: continue
                            
                            detection = {'bbox': [x1, y1, x2, y2], 'confidence': float(box.conf[0])}
                            if int(box.cls[0]) == 0: # Person
                                frame_detections['players'].append(detection)
                            elif int(box.cls[0]) == 32: # Sports Ball
                                if frame_detections['ball'] is None or detection['confidence'] > frame_detections['ball']['confidence']:# Prioritize higher confidence ball
                                    frame_detections['ball'] = detection
                
                all_detections.append({"frame_id": frame_idx, "detections": frame_detections})
                pbar.update(1)
        
        cap.release()
        with open(output_path, 'w') as f: json.dump(all_detections, f, indent=2)
        return all_detections

if __name__ == "__main__":
    detector = SoccerPlayerDetector()
    detector.process_video("9.mp4", "detections.json")
