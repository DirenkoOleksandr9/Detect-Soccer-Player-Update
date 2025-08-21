import cv2
import torch
import numpy as np
import json
import os
from ultralytics import YOLO
from tqdm import tqdm
from typing import List, Dict, Optional
import open_clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import pytesseract

# Force headless mode to avoid OpenGL issues
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'

class SoccerPlayerDetector:
    def __init__(self, model_name: str = 'yolov8l.pt', conf_thresh: float = 0.4, min_area: int = 500):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize YOLOv8-Large for player detection
        print(f"Loading YOLOv8-Large model for player detection...")
        self.player_model = YOLO('yolov8l.pt')
        self.player_model.to(self.device)
        
        # Initialize YOLOv8n for ball detection
        print(f"Loading YOLOv8n model for ball detection...")
        self.ball_model = YOLO('yolov8n.pt')
        self.ball_model.to(self.device)
        
        # Initialize OpenCLIP ViT-H/14 for Re-ID
        print(f"Loading OpenCLIP ViT-H/14 model for Re-ID...")
        self.reid_model, _, self.reid_preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k', device=self.device
        )
        self.reid_model.eval()
        
        self.conf_thresh = conf_thresh
        self.min_area = min_area
        self.target_classes = [0]  # Person only for player detection
        
        print(f"Enhanced Detector initialized on {self.device}")
        print(f"Player model: yolov8l.pt, Ball model: yolov8n.pt")
        print(f"Confidence threshold: {conf_thresh}, Min area: {min_area}")

    def get_embedding(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract OpenCLIP embeddings for Re-ID"""
        if crop_bgr.size == 0: 
            return None
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tensor = self.reid_preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                features = self.reid_model.encode_image(img_tensor)
                features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().squeeze()
        except Exception: 
            return None

    def get_color_hist(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract color histogram features"""
        if crop_bgr.size == 0: 
            return None
        try:
            lab_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist([lab_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except Exception: 
            return None

    def get_ssim(self, crop1_bgr: np.ndarray, crop2_bgr: np.ndarray) -> float:
        """Calculate structural similarity between two crops"""
        if crop1_bgr.size == 0 or crop2_bgr.size == 0: 
            return 0.0
        try:
            h, w, _ = crop1_bgr.shape
            crop2_resized = cv2.resize(crop2_bgr, (w, h))
            gray1 = cv2.cvtColor(crop1_bgr, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(crop2_resized, cv2.COLOR_BGR2GRAY)
            return ssim(gray1, gray2)
        except Exception: 
            return 0.0

    def ocr_jersey_number(self, crop_bgr: np.ndarray) -> Optional[str]:
        """Extract jersey number using OCR"""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 30 or crop_bgr.shape[1] < 30: 
            return None
        try:
            h, w, _ = crop_bgr.shape
            torso = crop_bgr[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (150, 75), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(gray_resized, (3,3), 0)
            sharpened = cv2.addWeighted(gray_resized, 1.5, blurred, -0.5, 0)
            _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = "--psm 8 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thresh, config=config, timeout=1).strip()
            return text if text.isdigit() else None
        except Exception: 
            return None

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
        
        with tqdm(total=total_frames, desc="Stage 1: Enhanced Detection & Feature Extraction") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: 
                    break
                
                # Player detection with YOLOv8-Large
                player_results = self.player_model(frame, classes=self.target_classes, verbose=False, conf=self.conf_thresh)
                
                # Ball detection with YOLOv8n
                ball_results = self.ball_model(frame, classes=[32], verbose=False, conf=0.5)
                
                frame_detections = {'players': [], 'ball': None}
                
                # Process player detections with enhanced features
                if len(player_results) > 0 and player_results[0].boxes is not None:
                    for box in player_results[0].boxes:
                        if box.conf[0] >= self.conf_thresh:
                            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                            area = (x2 - x1) * (y2 - y1)
                            
                            if area < self.min_area:
                                continue
                            
                            # Extract crop for feature extraction
                            crop = frame[y1:y2, x1:x2]
                            
                            # Extract all enhanced features
                            embedding = self.get_embedding(crop)
                            color_hist = self.get_color_hist(crop)
                            jersey_number = self.ocr_jersey_number(crop)
                            
                            if embedding is not None and color_hist is not None:
                                detection = {
                                    'bbox': [x1, y1, x2, y2], 
                                    'confidence': float(box.conf[0]),
                                    'area': area,
                                    'class': int(box.cls[0]),
                                    'embedding': embedding.tolist(),
                                    'color_hist': color_hist.tolist(),
                                    'jersey_number': jersey_number,
                                    'crop_data': crop.tolist() if crop.size < 10000 else None  # Limit crop size
                                }
                                frame_detections['players'].append(detection)
                
                # Process ball detection
                if len(ball_results) > 0 and ball_results[0].boxes is not None:
                    for box in ball_results[0].boxes:
                        if box.conf[0] >= 0.5:
                            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                            area = (x2 - x1) * (y2 - y1)
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2], 
                                'confidence': float(box.conf[0]),
                                'area': area,
                                'class': int(box.cls[0])
                            }
                            
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
        
        print(f"Enhanced detection complete. Saved to {output_path}")
        print(f"Total detections: {sum(len(frame['detections']['players']) for frame in all_detections)} players")
        print(f"Enhanced features: OpenCLIP embeddings, color histograms, jersey OCR, SSIM support")
        
        return all_detections

if __name__ == "__main__":
    detector = SoccerPlayerDetector()
    detector.process_video("9.mp4", "enhanced_detections.json")
