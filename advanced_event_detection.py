import cv2
import numpy as np
import json
from tqdm import tqdm
import math
from typing import List, Dict

class AdvancedEventDetector:
    def __init__(self, video_width: int, video_height: int, fps: float):
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps if fps > 0 else 30.0
        self.player_history = {}

        self.sprint_velocity_threshold = 20.0
        self.tackle_proximity_threshold = 75 
        self.fall_aspect_ratio_threshold = 1.4
        self.dribble_direction_change_threshold = 45
        self.goal_area = [video_width * 0.8, 0, video_width, video_height]
        self.celebration_cluster_size = 3
        self.celebration_cluster_radius = 150
        self.shot_to_celebration_window = 5 * self.fps

    def _update_player_history(self, players, frame_id):
        for p in players:
            pid = p['permanent_id']
            x1, y1, x2, y2 = p['bbox']
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1

            if pid not in self.player_history:
                self.player_history[pid] = []
            
            history = self.player_history[pid]
            velocity = np.array([0, 0])
            speed = 0
            if len(history) > 0:
                prev_center = history[-1]['center']
                velocity = center - prev_center
                speed = np.linalg.norm(velocity)

            history.append({
                'frame_id': frame_id,
                'center': center,
                'bbox': p['bbox'],
                'velocity': velocity,
                'speed': speed,
                'aspect_ratio': aspect_ratio
            })
            
            if len(history) > self.fps * 2:
                self.player_history[pid] = history[-int(self.fps * 2):]

    def detect_events(self, player_tracks_path: str, output_path: str) -> List[Dict]:
        with open(player_tracks_path, 'r') as f:
            all_tracks = json.load(f)
        
        events = []
        last_goal_area_entry = {}

        for frame_data in tqdm(all_tracks, desc="Stage 4: Detecting events"):
            frame_id = frame_data['frame_id']
            current_players = frame_data['players']
            
            self._update_player_history(current_players, frame_id)

            player_map = {p['permanent_id']: p for p in current_players}
            player_ids = list(player_map.keys())

            for pid in player_ids:
                history = self.player_history.get(pid, [])
                if not history: continue
                
                if history[-1]['speed'] > self.sprint_velocity_threshold:
                    events.append(self._create_event(frame_id, 'sprint', pid))
                
                if len(history) > 5:
                    v1 = history[-5]['velocity']
                    v2 = history[-1]['velocity']
                    angle = self._angle_between(v1, v2)
                    if angle > self.dribble_direction_change_threshold and history[-1]['speed'] > 5.0:
                        events.append(self._create_event(frame_id, 'skill_move', pid))
                
                center = history[-1]['center']
                if self.goal_area[0] < center[0] < self.goal_area[2] and self.goal_area[1] < center[1] < self.goal_area[3]:
                    last_goal_area_entry[pid] = frame_id

            for i in range(len(player_ids)):
                for j in range(i + 1, len(player_ids)):
                    pid1, pid2 = player_ids[i], player_ids[j]
                    hist1, hist2 = self.player_history.get(pid1), self.player_history.get(pid2)

                    if hist1 and hist2:
                        dist = np.linalg.norm(hist1[-1]['center'] - hist2[-1]['center'])
                        if dist < self.tackle_proximity_threshold:
                            if hist1[-1]['aspect_ratio'] > self.fall_aspect_ratio_threshold:
                                events.append(self._create_event(frame_id, 'tackle_fall', pid1))
                            if hist2[-1]['aspect_ratio'] > self.fall_aspect_ratio_threshold:
                                events.append(self._create_event(frame_id, 'tackle_fall', pid2))

            player_centers = [p_hist[-1]['center'] for pid, p_hist in self.player_history.items() if p_hist and p_hist[-1]['frame_id'] == frame_id]
            if len(player_centers) >= self.celebration_cluster_size:
                try:
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=self.celebration_cluster_radius, min_samples=self.celebration_cluster_size).fit(player_centers)
                    if len(set(clustering.labels_)) > 1:
                        for pid, entry_frame in last_goal_area_entry.items():
                            if frame_id - entry_frame < self.shot_to_celebration_window:
                                events.append(self._create_event(frame_id, 'goal_shot_attempt', pid))
                                last_goal_area_entry = {}
                                break
                except:
                    pass

        unique_events = self._deduplicate_events(events)
        with open(output_path, 'w') as f:
            json.dump(unique_events, f, indent=2)
        return unique_events

    def _create_event(self, frame_id, event_type, player_id=None):
        event = {
            'frame_id': frame_id,
            'timestamp': frame_id / self.fps,
            'event_type': event_type,
        }
        if player_id:
            event['player_id'] = player_id
        return event
    
    def _angle_between(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else np.array([0,0])
        v2_u = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else np.array([0,0])
        rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.rad2deg(rad)

    def _deduplicate_events(self, events):
        if not events:
            return []
        
        sorted_events = sorted(events, key=lambda x: x['frame_id'])
        
        unique_events = [sorted_events[0]]
        for current_event in sorted_events[1:]:
            last_event = unique_events[-1]
            if (current_event['frame_id'] - last_event['frame_id']) < self.fps * 2:
                continue
            unique_events.append(current_event)
            
        return unique_events

    def filter_player_events(self, events_path: str, target_player_id: int, output_path: str) -> List[Dict]:
        with open(events_path, 'r') as f:
            all_events = json.load(f)
        
        player_events = [e for e in all_events if e.get('player_id') == target_player_id or 'cluster' in e.get('event_type', '') or 'goal' in e.get('event_type', '')]
        
        with open(output_path, 'w') as f:
            json.dump(player_events, f, indent=2)
        
        return player_events
