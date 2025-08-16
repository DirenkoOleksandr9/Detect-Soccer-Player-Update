import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
from tqdm import tqdm
import os

class SoccerPlayerDetector:
    def __init__(self, model_path=None, conf_thresh: float = 0.35, min_area: int = 1000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        self.conf_thresh = conf_thresh
        self.min_area = min_area
        
    def detect_frame(self, frame):
        results = self.model(frame, classes=[0], imgsz=960, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    if conf >= self.conf_thresh:
                        if (x2 - x1) * (y2 - y1) < self.min_area:
                            continue
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': 0,
                            'class_name': 'person'
                        })
        return detections
    
    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Detecting players") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = self.detect_frame(frame)
                frame_data = {
                    "frame_id": frame_idx,
                    "detections": detections
                }
                all_detections.append(frame_data)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        with open(output_path, 'w') as f:
            json.dump(all_detections, f)
        
        return all_detections

if __name__ == "__main__":
    detector = SoccerPlayerDetector()
    detector.process_video("9.mp4", "detections.json")
