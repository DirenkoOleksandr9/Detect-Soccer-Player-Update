import cv2
import torch
import numpy as np
import json
import easyocr
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import os

class OSNetFeatureExtractor:
    def __init__(self, model_name='osnet_ain_x1_0', device='cuda'):
        self.device = device
        self.model = self._load_osnet_model(model_name)
        self.model.eval()
        self.model.to(device)
        
    def _load_osnet_model(self, model_name):
        try:
            import torchreid
            model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                loss='softmax',
                pretrained=True
            )
            return model
        except ImportError:
            print("Warning: torchreid not available, using fallback CNN")
            return self._create_fallback_cnn()
    
    def _create_fallback_cnn(self):
        class FallbackCNN(nn.Module):
            def __init__(self, embedding_dim=512):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(128)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(256)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, embedding_dim)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.bn3(self.conv3(x)))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)
        
        return FallbackCNN()
    
    def extract_features(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
            if isinstance(features, tuple):
                features = features[0]
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy().flatten()

class AdvancedPlayerReID:
    def __init__(self, similarity_threshold=0.6, jersey_bonus=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = OSNetFeatureExtractor(device=self.device)
        self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        self.global_players = {}
        self.next_permanent_id = 1
        self.similarity_threshold = similarity_threshold
        self.jersey_bonus = jersey_bonus
        self.ocr_validation_buffer = 5
        
        print(f"Advanced Re-ID initialized on {self.device}")
        print(f"Similarity threshold: {similarity_threshold}, Jersey bonus: {jersey_bonus}")
    
    def get_features(self, patch):
        if patch.size == 0:
            return None
            
        try:
            patch_resized = cv2.resize(patch, (256, 256))
            
            # Deep features using OSNet
            deep_features = self.feature_extractor.extract_features(patch_resized)
            
            # Color histogram
            hsv = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2HSV)
            color_hist = cv2.normalize(
                cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]), 
                None
            ).flatten()
            
            # Dominant color
            pixels = cv2.resize(patch, (10, 10)).reshape(-1, 3)
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # Jersey number OCR
            jersey = None
            try:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                results = self.ocr.readtext(
                    gray, 
                    allowlist='0123456789', 
                    detail=0, 
                    paragraph=False,
                    height_ths=0.5,
                    width_ths=0.5
                )
                if results and results[0].isdigit() and 1 <= len(results[0]) <= 2:
                    jersey = int(results[0])
            except Exception:
                pass
            
            return {
                'deep': deep_features,
                'color': color_hist,
                'dominant_colors': dominant_colors,
                'jersey': jersey
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0
        
        # Deep feature similarity (primary)
        deep_sim = cosine_similarity(
            features1['deep'].reshape(1, -1), 
            features2['deep'].reshape(1, -1)
        )[0][0]
        
        # Color histogram similarity
        color_sim = cv2.compareHist(
            features1['color'], 
            features2['color'], 
            cv2.HISTCMP_CORREL
        )
        
        # Dominant color similarity
        color_dist = np.linalg.norm(
            features1['dominant_colors'].mean(axis=0) - 
            features2['dominant_colors'].mean(axis=0)
        )
        color_sim_2 = max(0, 1 - color_dist / 255)
        
        # Combined similarity
        total_sim = 0.7 * deep_sim + 0.2 * color_sim + 0.1 * color_sim_2
        
        # Jersey number bonus
        if (features1['jersey'] is not None and 
            features2['jersey'] is not None and 
            features1['jersey'] == features2['jersey']):
            total_sim += self.jersey_bonus
        
        return total_sim
    
    def process_tracklets(self, tracklets_path, video_path, output_path):
        with open(tracklets_path, 'r') as f:
            all_tracklets = json.load(f)
        
        cap = cv2.VideoCapture(video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        long_tracks = []
        all_player_colors = []
        
        print(f"Processing {len(all_tracklets)} frames for Re-ID")
        
        for frame_data in tqdm(all_tracklets, desc="Stage 3: Advanced Re-ID"):
            frame_id = frame_data['frame_id']
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_tracks = {"frame_id": frame_id, "players": []}
            
            for track in frame_data['tracks']:
                x1, y1, x2, y2 = track['bbox']
                
                # Clamp coordinates to prevent out-of-bounds
                x1 = max(0, min(frame_w - 1, int(x1)))
                y1 = max(0, min(frame_h - 1, int(y1)))
                x2 = max(0, min(frame_w, int(x2)))
                y2 = max(0, min(frame_h, int(y2)))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                current_features = self.get_features(patch)
                if current_features is None:
                    continue
                
                best_id = None
                best_score = self.similarity_threshold
                
                # Multi-stage matching
                for pid, p_info in self.global_players.items():
                    # Stage 1: High-confidence jersey match
                    if (p_info.get('confirmed_jersey') and 
                        current_features['jersey'] == p_info['confirmed_jersey']):
                        best_id = pid
                        best_score = 1.0
                        break
                    
                    # Stage 2: Appearance similarity
                    sim = self.calculate_similarity(current_features, p_info['features'])
                    if sim > best_score:
                        best_score = sim
                        best_id = pid
                
                # Assign new ID if no match found
                if best_id is None:
                    best_id = self.next_permanent_id
                    self.next_permanent_id += 1
                    self.global_players[best_id] = {
                        'features': current_features,
                        'ocr_buffer': [],
                        'last_seen': frame_id
                    }
                else:
                    # Update existing player
                    p_info = self.global_players[best_id]
                    p_info['last_seen'] = frame_id
                    
                    # Update features with exponential moving average
                    alpha = 0.1
                    p_info['features']['deep'] = (
                        (1 - alpha) * p_info['features']['deep'] + 
                        alpha * current_features['deep']
                    )
                    p_info['features']['color'] = (
                        (1 - alpha) * p_info['features']['color'] + 
                        alpha * current_features['color']
                    )
                
                # Update OCR buffer
                if best_id in self.global_players:
                    self.global_players[best_id]['ocr_buffer'].append(current_features['jersey'])
                    if len(self.global_players[best_id]['ocr_buffer']) > 10:
                        self.global_players[best_id]['ocr_buffer'].pop(0)
                    
                    # Validate jersey number
                    ocr_counts = Counter([
                        j for j in self.global_players[best_id]['ocr_buffer'] 
                        if j is not None
                    ])
                    if ocr_counts and ocr_counts.most_common(1)[0][1] >= self.ocr_validation_buffer:
                        self.global_players[best_id]['confirmed_jersey'] = ocr_counts.most_common(1)[0][0]
                
                # Collect colors for team classification
                if current_features['dominant_colors'] is not None:
                    all_player_colors.append(current_features['dominant_colors'].mean(axis=0))
                
                frame_tracks["players"].append({
                    "permanent_id": best_id,
                    "bbox": [x1, y1, x2, y2],
                    "jersey": self.global_players[best_id].get('confirmed_jersey'),
                    "confidence": track.get('confidence', 0.0)
                })
            
            long_tracks.append(frame_tracks)
        
        cap.release()
        
        # Team classification using K-means
        if len(all_player_colors) > 10:
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(all_player_colors)
                
                for pid, p_info in self.global_players.items():
                    if 'features' in p_info and p_info['features']['dominant_colors'] is not None:
                        p_color = p_info['features']['dominant_colors'].mean(axis=0).reshape(1, -1)
                        p_info['team'] = int(kmeans.predict(p_color)[0])
                
                # Assign team to frame data
                for frame_data in long_tracks:
                    for player in frame_data['players']:
                        pid = player['permanent_id']
                        if pid in self.global_players and 'team' in self.global_players[pid]:
                            player['team'] = self.global_players[pid]['team']
            except Exception as e:
                print(f"Team classification failed: {e}")
        
        with open(output_path, 'w') as f:
            json.dump(long_tracks, f, indent=2)
        
        print(f"Re-ID complete. Saved to {output_path}")
        print(f"Total unique players: {len(self.global_players)}")
        print(f"Players with confirmed jerseys: {sum(1 for p in self.global_players.values() if p.get('confirmed_jersey'))}")
        
        return long_tracks

if __name__ == "__main__":
    reid = AdvancedPlayerReID()
    reid.process_tracklets("tracklets.json", "9.mp4", "long_player_track.json")
