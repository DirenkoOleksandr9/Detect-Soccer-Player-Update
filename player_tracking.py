
# pip install -U ultralytics easyocr
# Copyright © 2025 Oleksandr Direnko — Soccer Player Detecting System

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
import easyocr
import math
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Feature Extractor Network
# ----------------------------
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


# ----------------------------
# Player Tracker Class
# ----------------------------
class PlayerTracker:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(model_path)
        self.feature_extractor = PlayerFeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        self.next_id = 1
        self.players = {}
        self.positions = {}


    # Getting Color Features by converting from rgb to HSV image
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

    
    # Getting Features using Deep Learning CNN (Zero - shot)
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

    # Trying to get Jersey Number for a better tracking
    def read_jersey_number(self, patch):
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Using EasyOCR here
            results = self.ocr.readtext(gray, allowlist='0123456789')

            for _, text, conf in results:
                if text.isdigit() and 1 <= len(text) <= 2:
                    return int(text), conf
        except:
            pass
        return None, 0.0

    # Gives hypotenius
    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # Match Players with the help of Previous Position and features
    def match_player(self, current_features, current_position, assigned_ids):
        best_id = None
        best_score = 0.65

        for player_id, info in self.players.items():
            if player_id in assigned_ids:
                continue

            prev_features = info['features']
            prev_position = self.positions.get(player_id)

            if prev_position and self.distance(current_position, prev_position) > 250:
                continue

            color_sim = cosine_similarity(
                current_features['color'].reshape(1, -1),
                prev_features['color'].reshape(1, -1)
            )[0][0]

            deep_sim = cosine_similarity(
                current_features['deep'].reshape(1, -1),
                prev_features['deep'].reshape(1, -1)
            )[0][0]

            score = (color_sim + deep_sim) / 2

            j1 = current_features.get('jersey')
            j2 = prev_features.get('jersey')
            if j1 is not None and j2 is not None and j1 == j2:
                score += 0.15

            if score > best_score:
                best_score = score
                best_id = player_id

        return best_id

    # Update the player in dictionary
    def update_player(self, player_id, features, position):
        if player_id not in self.players:
            self.players[player_id] = {'features': features, 'count': 1}
        else:
            old_feat = self.players[player_id]['features']
            old_feat['color'] = (old_feat['color'] + features['color']) / 2
            old_feat['deep'] = (old_feat['deep'] + features['deep']) / 2
            if features.get('jersey') is not None:
                old_feat['jersey'] = features['jersey']
            self.players[player_id]['count'] += 1

        self.positions[player_id] = position

    # Drawing a Rectangle around the player
    def draw_player(self, frame, bbox, player_id, jersey, color=(0, 255, 0)):
        if(player_id == 8):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {player_id}"
            if jersey is not None:
                label += f" #{jersey}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    # Processing the given frame
    def process(self, frame):
        results = self.yolo(frame)[0]
        assigned_ids = set()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf < 0.4 or (x2 - x1) < 30 or (y2 - y1) < 50:
                continue

            patch = frame[y1:y2, x1:x2]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            features = {
                'color': self.get_color_features(patch),
                'deep': self.get_deep_features(patch)
            }

            jersey, _ = self.read_jersey_number(patch)
            features['jersey'] = jersey

            matched_id = self.match_player(features, center, assigned_ids)
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            assigned_ids.add(matched_id)
            self.update_player(matched_id, features, center)
            self.draw_player(frame, (x1, y1, x2, y2), matched_id, jersey)

        return frame


# ----------------------------
# Main Processing Function
# ----------------------------
def main():
    model_path = "./best.pt"
    video_path = "./9.mp4"
    output_path = "tracked_players_output.mp4"

    print(f"[INFO] Loading YOLO model from: {model_path}")
    print(f"[INFO] Input video: {video_path}")

    tracker = PlayerTracker(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Failed to open the video file.")
        return

    # Capturing specs from the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # Store the Video frame wise

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_frame = tracker.process(frame) # Will call for the above PlayerTracker Class
        out.write(tracked_frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[FRAME {frame_idx}/{total_frames}] Players tracked: {len(tracker.players)}")

    cap.release()
    out.release()

    print(f"\n[INFO] Processing complete.")
    print(f"[INFO] Frames processed: {frame_idx}")
    print(f"[INFO] Unique players: {len(tracker.players)}")
    print(f"[INFO] Output saved to: {output_path}")


# Run the main function
if __name__ == "__main__":
    main()
