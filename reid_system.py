import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import easyocr
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math

class PlayerFeatureExtractor(nn.Module):
    def __init__(self):
        super(PlayerFeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.feature_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        features = self.feature_head(x)
        return F.normalize(features, p=2, dim=1)

class PlayerReID:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = PlayerFeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.global_players = {}
        self.next_permanent_id = 1
        self.similarity_threshold = 0.7
        self.jersey_bonus = 0.3

    def get_color_features(self, patch):
        try:
            resized = cv2.resize(patch, (64, 64))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            h = cv2.normalize(h, h).flatten()
            s = cv2.normalize(s, s).flatten()
            v = cv2.normalize(v, v).flatten()
            return np.concatenate([h, s, v])
        except:
            return np.zeros(94)

    def get_deep_features(self, patch):
        try:
            img = cv2.resize(patch, (224, 224))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
            return features.cpu().numpy().flatten()
        except:
            return np.zeros(128)

    def read_jersey_number(self, patch):
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            results = self.ocr.readtext(gray, allowlist='0123456789')
            for _, text, conf in results:
                if text.isdigit() and 1 <= len(text) <= 2:
                    return int(text), conf
        except:
            pass
        return None, 0.0

    def calculate_similarity(self, features1, features2):
        color_sim = cosine_similarity(
            features1['color'].reshape(1, -1),
            features2['color'].reshape(1, -1)
        )[0][0]
        deep_sim = cosine_similarity(
            features1['deep'].reshape(1, -1),
            features2['deep'].reshape(1, -1)
        )[0][0]
        return (color_sim + deep_sim) / 2

    def match_player(self, current_features, current_position):
        best_id = None
        best_score = self.similarity_threshold

        for player_id, info in self.global_players.items():
            prev_features = info['features']
            prev_position = info.get('last_position')
            
            if prev_position and self.distance(current_position, prev_position) > 300:
                continue

            similarity = self.calculate_similarity(current_features, prev_features)
            
            j1 = current_features.get('jersey')
            j2 = prev_features.get('jersey')
            if j1 is not None and j2 is not None and j1 == j2:
                similarity += self.jersey_bonus

            if similarity > best_score:
                best_score = similarity
                best_id = player_id

        return best_id

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update_player(self, player_id, features, position, frame_id):
        if player_id not in self.global_players:
            self.global_players[player_id] = {
                'features': features,
                'last_position': position,
                'last_seen': frame_id,
                'count': 1
            }
        else:
            old_feat = self.global_players[player_id]['features']
            old_feat['color'] = (old_feat['color'] + features['color']) / 2
            old_feat['deep'] = (old_feat['deep'] + features['deep']) / 2
            if features.get('jersey') is not None:
                old_feat['jersey'] = features['jersey']
            self.global_players[player_id]['last_position'] = position
            self.global_players[player_id]['last_seen'] = frame_id
            self.global_players[player_id]['count'] += 1

    def process_tracklets(self, tracklets_path, video_path, output_path):
        with open(tracklets_path, 'r') as f:
            all_tracklets = json.load(f)
        
        cap = cv2.VideoCapture(video_path)
        long_tracks = []
        
        for frame_data in tqdm(all_tracklets, desc="Re-identifying players"):
            frame_id = frame_data['frame_id']
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_tracks = {
                "frame_id": frame_id,
                "players": []
            }
            
            for track in frame_data['tracks']:
                x1, y1, x2, y2 = track['bbox']
                patch = frame[y1:y2, x1:x2]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                features = {
                    'color': self.get_color_features(patch),
                    'deep': self.get_deep_features(patch)
                }
                
                jersey, _ = self.read_jersey_number(patch)
                features['jersey'] = jersey
                
                matched_id = self.match_player(features, center)
                if matched_id is None:
                    matched_id = self.next_permanent_id
                    self.next_permanent_id += 1
                
                self.update_player(matched_id, features, center, frame_id)
                
                frame_tracks["players"].append({
                    "permanent_id": matched_id,
                    "bbox": track['bbox'],
                    "confidence": track['confidence'],
                    "jersey": jersey
                })
            
            long_tracks.append(frame_tracks)
        
        cap.release()
        
        with open(output_path, 'w') as f:
            json.dump(long_tracks, f)
        
        return long_tracks

if __name__ == "__main__":
    reid = PlayerReID()
    reid.process_tracklets("tracklets.json", "9.mp4", "long_player_track.json")
