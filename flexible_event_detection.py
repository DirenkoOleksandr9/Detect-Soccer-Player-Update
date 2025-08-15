import cv2
import json
import math
from tqdm import tqdm
from typing import List, Dict

class AdaptiveEventDetector:
    """Detects events using adaptive thresholds and multiple detection strategies."""
    def __init__(self, video_width: int, video_height: int):
        self.player_history: Dict[int, List] = {}
        self.goal_area = [video_width * 0.85, video_height * 0.2, video_width, video_height * 0.8]
        self.goal_area_left = [0, video_height * 0.2, video_width * 0.15, video_height * 0.8]
        
        # Much more flexible thresholds
        self.velocity_threshold = 5.0  # Very sensitive to movement
        self.cluster_threshold = 60    # Smaller clusters
        self.cluster_size_threshold = 2 # Just 2 players needed
        self.movement_threshold = 2.0   # Detect any movement
        
        # Field analysis
        self.field_center = (video_width // 2, video_height // 2)
        self.field_radius = min(video_width, video_height) // 4
        
    def detect_events(self, player_tracks_path: str, video_path: str, output_path: str) -> List[Dict]:
        with open(player_tracks_path, 'r') as f: all_tracks = json.load(f)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        events = []
        
        print(f"🎯 Processing {len(all_tracks)} frames for events...")
        
        for frame_data in tqdm(all_tracks, desc="Stage 4: Detecting events"):
            frame_id = frame_data['frame_id']
            current_players = frame_data['players']
            
            if len(current_players) == 0:
                continue
                
            player_centers = {}
            
            for p in current_players:
                pid = p['permanent_id']
                x1, y1, x2, y2 = p['bbox']
                center = ((x1+x2)/2, (y1+y2)/2)
                player_centers[pid] = center
                
                # Event 1: Any movement detection (very sensitive)
                if pid in self.player_history and len(self.player_history[pid]) > 1:
                    prev_center = self.player_history[pid][-2]['center']
                    velocity = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                    
                    # Detect ANY movement
                    if velocity > self.movement_threshold:
                        events.append({
                            'frame_id': frame_id, 
                            'timestamp': frame_id / fps, 
                            'event_type': 'player_movement', 
                            'player_id': pid
                        })
                    
                    # Detect sprints
                    if velocity > self.velocity_threshold:
                        events.append({
                            'frame_id': frame_id, 
                            'timestamp': frame_id / fps, 
                            'event_type': 'sprint', 
                            'player_id': pid
                        })
                
                # Event 2: Goal area proximity (both sides)
                if (self.goal_area[0] < center[0] < self.goal_area[2] and 
                    self.goal_area[1] < center[1] < self.goal_area[3]):
                    events.append({
                        'frame_id': frame_id, 
                        'timestamp': frame_id / fps, 
                        'event_type': 'goal_area_entry_right', 
                        'player_id': pid
                    })
                
                if (self.goal_area_left[0] < center[0] < self.goal_area_left[2] and 
                    self.goal_area_left[1] < center[1] < self.goal_area_left[3]):
                    events.append({
                        'frame_id': frame_id, 
                        'timestamp': frame_id / fps, 
                        'event_type': 'goal_area_entry_left', 
                        'player_id': pid
                    })
                
                # Event 3: Field center activity
                distance_from_center = math.hypot(center[0] - self.field_center[0], center[1] - self.field_center[1])
                if distance_from_center < self.field_radius:
                    events.append({
                        'frame_id': frame_id, 
                        'timestamp': frame_id / fps, 
                        'event_type': 'midfield_action', 
                        'player_id': pid
                    })
                
                # Event 4: Edge of screen activity
                edge_threshold = 30
                if (center[0] < edge_threshold or center[0] > video_width - edge_threshold or
                    center[1] < edge_threshold or center[1] > video_height - edge_threshold):
                    events.append({
                        'frame_id': frame_id, 
                        'timestamp': frame_id / fps, 
                        'event_type': 'wide_play', 
                        'player_id': pid
                    })
                
                # Update history
                if pid not in self.player_history: 
                    self.player_history[pid] = []
                self.player_history[pid].append({'frame': frame_id, 'center': center})
            
            # Event 5: Player clustering (very sensitive)
            if len(player_centers) >= self.cluster_size_threshold:
                try:
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=self.cluster_threshold, min_samples=2).fit(list(player_centers.values()))
                    if len(set(clustering.labels_)) > 1:
                        events.append({
                            'frame_id': frame_id, 
                            'timestamp': frame_id / fps, 
                            'event_type': 'player_cluster',
                            'cluster_size': len(player_centers)
                        })
                except:
                    pass
            
            # Event 6: High player density
            if len(current_players) >= 6:
                events.append({
                    'frame_id': frame_id, 
                    'timestamp': frame_id / fps, 
                    'event_type': 'high_density',
                    'player_count': len(current_players)
                })
        
        cap.release()
        
        # Ensure we have events even if detection was sparse
        if len(events) == 0:
            print("⚠️ No events detected. Creating fallback events...")
            events = self.create_fallback_events(all_tracks, fps)
        
        # Deduplicate with flexible timing
        unique_events = self.deduplicate_events(events, fps)
        
        with open(output_path, 'w') as f: 
            json.dump(unique_events, f, indent=2)
        
        print(f"🎯 Detected {len(unique_events)} events with flexible thresholds")
        return unique_events
    
    def create_fallback_events(self, all_tracks: List[Dict], fps: float) -> List[Dict]:
        """Create fallback events when normal detection fails."""
        events = []
        
        # Create events every 3 seconds as fallback
        event_interval = int(3 * fps)
        
        for i, frame_data in enumerate(all_tracks):
            if i % event_interval == 0 and i > 0:
                events.append({
                    'frame_id': frame_data['frame_id'],
                    'timestamp': frame_data['frame_id'] / fps,
                    'event_type': 'periodic_highlight',
                    'player_id': 1
                })
        
        # Add start and end events
        if all_tracks:
            events.append({
                'frame_id': 0,
                'timestamp': 0.0,
                'event_type': 'match_start',
                'player_id': 1
            })
            
            events.append({
                'frame_id': all_tracks[-1]['frame_id'],
                'timestamp': all_tracks[-1]['frame_id'] / fps,
                'event_type': 'match_end',
                'player_id': 1
            })
        
        return events
    
    def deduplicate_events(self, events: List[Dict], fps: float) -> List[Dict]:
        """Deduplicate events with flexible timing (1.5-second window)."""
        if not events:
            return events
            
        unique_events = []
        seen_times = set()
        time_window = 1.5  # 1.5-second window for deduplication
        
        for event in sorted(events, key=lambda x: x['timestamp']):
            event_time = event['timestamp']
            
            # Check if we have a similar event within the time window
            is_duplicate = False
            for seen_time in seen_times:
                if abs(event_time - seen_time) < time_window:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_events.append(event)
                seen_times.add(event_time)
        
        return unique_events

def filter_player_events(events_path: str, target_player_id: int, output_path: str) -> List[Dict]:
    """Filter events for target player with fallback to general events."""
    with open(events_path, 'r') as f: 
        all_events = json.load(f)
    
    # First, try to get player-specific events
    player_events = [e for e in all_events if e.get('player_id') == target_player_id]
    
    # If no player-specific events, include general events
    if len(player_events) == 0:
        print(f"⚠️ No events found for player {target_player_id}. Including general events...")
        general_events = [e for e in all_events if e['event_type'] in [
            'player_cluster', 'high_density', 'periodic_highlight', 'match_start', 'match_end'
        ]]
        player_events = general_events
    
    # Ensure we have at least some events
    if len(player_events) == 0:
        print("⚠️ Still no events. Creating basic timeline events...")
        player_events = [{'frame_id': 0, 'timestamp': 0.0, 'event_type': 'basic_highlight', 'player_id': target_player_id}]
    
    with open(output_path, 'w') as f: 
        json.dump(player_events, f, indent=2)
    
    return player_events
