import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import json
from tqdm import tqdm
import math

class SlowFastNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(SlowFastNetwork, self).__init__()
        self.slow_path = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.fast_path = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(8, 8, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(192 + 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, slow_input, fast_input):
        slow_features = self.slow_path(slow_input)
        fast_features = self.fast_path(fast_input)
        
        slow_features = slow_features.view(slow_features.size(0), -1)
        fast_features = fast_features.view(fast_features.size(0), -1)
        
        combined_features = torch.cat([slow_features, fast_features], dim=1)
        output = self.classifier(combined_features)
        return output

class EventDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SlowFastNetwork(num_classes=5)
        self.model.to(self.device)
        self.model.eval()
        
        self.event_classes = ['goal', 'shot', 'pass', 'tackle', 'normal']
        self.clip_length = 16
        self.slow_stride = 16
        self.fast_stride = 4
        self.confidence_threshold = 0.6

    def preprocess_clip(self, frames):
        if len(frames) < self.clip_length:
            return None, None
        
        slow_frames = frames[::self.slow_stride][:self.clip_length//self.slow_stride]
        fast_frames = frames[::self.fast_stride][:self.clip_length]
        
        slow_input = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in slow_frames])
        fast_input = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in fast_frames])
        
        slow_input = slow_input.unsqueeze(0).to(self.device)
        fast_input = fast_input.unsqueeze(0).to(self.device)
        
        return slow_input, fast_input

    def detect_events(self, video_path, player_tracks_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with open(player_tracks_path, 'r') as f:
            player_tracks = json.load(f)
        
        events = []
        frame_buffer = []
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Detecting events") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_buffer.append(frame)
                if len(frame_buffer) > self.clip_length:
                    frame_buffer.pop(0)
                
                if len(frame_buffer) == self.clip_length and frame_idx % 8 == 0:
                    slow_input, fast_input = self.preprocess_clip(frame_buffer)
                    
                    if slow_input is not None:
                        with torch.no_grad():
                            output = self.model(slow_input, fast_input)
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            if confidence.item() > self.confidence_threshold:
                                event_type = self.event_classes[predicted.item()]
                                if event_type != 'normal':
                                    timestamp = frame_idx / fps
                                    events.append({
                                        'frame_id': frame_idx,
                                        'timestamp': timestamp,
                                        'event_type': event_type,
                                        'confidence': confidence.item()
                                    })
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        with open(output_path, 'w') as f:
            json.dump(events, f)
        
        return events

    def filter_player_events(self, events_path, player_tracks_path, target_player_id, output_path):
        with open(events_path, 'r') as f:
            all_events = json.load(f)
        
        with open(player_tracks_path, 'r') as f:
            player_tracks = json.load(f)
        
        player_events = []
        
        for event in all_events:
            frame_id = event['frame_id']
            
            if frame_id < len(player_tracks):
                frame_players = player_tracks[frame_id]['players']
                
                for player in frame_players:
                    if player['permanent_id'] == target_player_id:
                        player_events.append({
                            'frame_id': frame_id,
                            'timestamp': event['timestamp'],
                            'event_type': event['event_type'],
                            'confidence': event['confidence'],
                            'player_id': target_player_id
                        })
                        break
        
        with open(output_path, 'w') as f:
            json.dump(player_events, f)
        
        return player_events

if __name__ == "__main__":
    detector = EventDetector()
    detector.detect_events("9.mp4", "long_player_track.json", "events.json")
    detector.filter_player_events("events.json", "long_player_track.json", 1, "player_events.json")
