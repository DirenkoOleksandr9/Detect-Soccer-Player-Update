"""
Cerebrus Tracking System v9 - Continuity-Guided Re-Identification
Enhanced single-player tracking with ball detection and robust Re-ID
"""

import cv2
import numpy as np
import json
from scipy.spatial.distance import cdist
import collections
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm

def iou(boxA, boxB):
    """Calculate Intersection over Union between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val if not np.isnan(iou_val) else 0.0

class CerebrusTrack:
    """Enhanced tracking object with multiple identification methods"""
    
    def __init__(self, bbox, track_id, data, is_manual_init=False):
        self.id = track_id
        self.bbox = np.array(bbox)
        self.time_since_update = 0
        self.hits = 1
        self.state = 'confirmed' if is_manual_init else 'tentative'
        
        # Enhanced feature storage
        self.feature_ema = data['embedding']
        self.color_hist_ema = data['color_hist']
        self.last_known_crop = data.get('crop_data')
        self.jersey_number = data.get('jersey_number')
        self.confirmed_number = self.jersey_number if is_manual_init else None
        
        # Jersey number validation buffer
        self.number_hits = collections.defaultdict(int)
        if self.jersey_number: 
            self.number_hits[self.jersey_number] += 5 if is_manual_init else 1

    def update(self, bbox, data):
        """Update track with new detection data"""
        self.time_since_update = 0
        self.hits += 1
        self.bbox = np.array(bbox)
        
        # State management
        if self.state == 'lost': 
            self.state = 'confirmed'
        if self.hits > 5 and self.state == 'tentative': 
            self.state = 'confirmed'
        
        # Exponential Moving Average updates
        self.feature_ema = 0.9 * self.feature_ema + 0.1 * data['embedding']
        self.color_hist_ema = 0.9 * self.color_hist_ema + 0.1 * data['color_hist']
        self.last_known_crop = data.get('crop_data')
        
        # Jersey number validation
        if data.get('jersey_number'):
            self.number_hits[data['jersey_number']] += 1
            if self.number_hits[data['jersey_number']] > 5: 
                self.confirmed_number = data['jersey_number']

class CerebrusTracker:
    """Main tracking system with Continuity-Guided Re-ID"""
    
    def __init__(self, max_age=90):
        self.max_age = max_age
        
        # Enhanced thresholds for robust tracking
        self.reid_thresh = 0.7
        self.color_thresh = 0.6
        self.ssim_thresh = 0.2
        
        self.track_id_counter = 1
        self.target_track = None
        self.ball_pos = None
        
        print(f"Cerebrus Tracker v9 initialized with max_age={max_age}")

    def initialize_target(self, initial_data):
        """Initialize target tracking with manual selection"""
        self.target_track = CerebrusTrack(
            initial_data['bbox'], 
            self.track_id_counter, 
            initial_data, 
            is_manual_init=True
        )
        self.track_id_counter += 1
        print(f"Target initialized: Track ID {self.target_track.id}")

    def update(self, frame, p_data, b_detections, target_jersey_number=None):
        """Update tracking with new frame data"""
        
        # Update ball position
        if len(b_detections) > 0:
            best_ball = b_detections[np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in b_detections])]
            self.ball_pos = (int((best_ball[0]+best_ball[2])/2), int((best_ball[1]+best_ball[3])/2))

        # Initialize target if not exists
        if not self.target_track and target_jersey_number is not None:
            if p_data:
                found_player = None
                if target_jersey_number:
                    # Find player by jersey number
                    for d in p_data:
                        if d.get('jersey_number') == target_jersey_number: 
                            found_player = d
                            break
                else:
                    # Find most central player
                    pitch_center_x = frame.shape[1] / 2
                    centrality = [abs((d['bbox'][0]+d['bbox'][2])/2 - pitch_center_x) for d in p_data]
                    found_player = p_data[np.argmin(centrality)]
                
                if found_player: 
                    self.target_track = CerebrusTrack(
                        found_player['bbox'], 
                        self.track_id_counter, 
                        found_player
                    )
                    self.track_id_counter += 1

        # Return early if no target
        if not self.target_track: 
            return [], self.ball_pos

        # Update target age
        self.target_track.time_since_update += 1
        if self.target_track.time_since_update > self.max_age:
            self.target_track = None
            return [], self.ball_pos
        
        # Track matching logic
        match_found = False
        if p_data:
            last_box = self.target_track.bbox
            w, h = last_box[2] - last_box[0], last_box[3] - last_box[1]
            
            # Define generous search area based on player size
            search_area = [
                last_box[0] - w*1.5, 
                last_box[1] - h*1.5, 
                last_box[2] + w*1.5, 
                last_box[3] + h*1.5
            ]
            
            # Find candidates in search area
            candidate_indices = [i for i, d in enumerate(p_data) if iou(d['bbox'], search_area) > 0]
            
            if candidate_indices:
                candidate_detections = [p_data[i] for i in candidate_indices]
                
                # Calculate multiple similarity metrics
                reid_dists = cdist(
                    np.array([self.target_track.feature_ema]), 
                    np.array([d['embedding'] for d in candidate_detections]), 
                    'cosine'
                ).flatten()
                
                color_corrs = np.array([
                    cv2.compareHist(
                        self.target_track.color_hist_ema, 
                        d['color_hist'], 
                        cv2.HISTCMP_CORREL
                    ) for d in candidate_detections
                ])
                
                # SSIM scores (if crop data available)
                ssim_scores = np.array([
                    self._calculate_ssim(self.target_track.last_known_crop, d.get('crop_data')) 
                    for d in candidate_detections
                ])
                
                # --- Continuity Score Calculation ---
                last_center = np.array([
                    (last_box[0] + last_box[2]) / 2, 
                    (last_box[1] + last_box[3]) / 2
                ])
                candidate_centers = np.array([
                    ((d['bbox'][0] + d['bbox'][2]) / 2, (d['bbox'][1] + d['bbox'][3]) / 2) 
                    for d in candidate_detections
                ])
                
                # Normalize distance by player height for scale-invariance
                distances = np.linalg.norm(candidate_centers - last_center, axis=1) / h
                continuity_scores = np.exp(-distances)  # Closer = higher score

                # --- Intelligent Fused Score ---
                # Lower is better, heavily weight continuity
                fused_scores = (reid_dists) + (1 - color_corrs) + (1 - ssim_scores) - (continuity_scores * 0.5)
                
                best_candidate_idx = np.argmin(fused_scores)
                
                # Use holistic threshold for decision
                if fused_scores[best_candidate_idx] < 1.5:
                    original_idx = candidate_indices[best_candidate_idx]
                    d = p_data[original_idx]
                    self.target_track.update(d['bbox'], d)
                    match_found = True

        # Update state if no match found
        if not match_found:
            self.target_track.state = 'lost'

        # Return output tracks
        output_tracks = [self.target_track] if self.target_track and self.target_track.state != 'tentative' else []
        return output_tracks, self.ball_pos

    def _calculate_ssim(self, crop1, crop2):
        """Calculate SSIM between two crops"""
        if crop1 is None or crop2 is None:
            return 0.0
        try:
            # Convert back to numpy arrays if they were stored as lists
            if isinstance(crop1, list):
                crop1 = np.array(crop1)
            if isinstance(crop2, list):
                crop2 = np.array(crop2)
            
            if crop1.size == 0 or crop2.size == 0:
                return 0.0
                
            h, w, _ = crop1.shape
            crop2_resized = cv2.resize(crop2, (w, h))
            gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(crop2_resized, cv2.COLOR_BGR2GRAY)
            
            from skimage.metrics import structural_similarity as ssim
            return ssim(gray1, gray2)
        except Exception:
            return 0.0

def process_video_with_cerebrus(
    input_path: str, 
    output_path: str, 
    highlight_path: str, 
    mode: str = 'Automatic',
    initial_target_data: Optional[Dict[str, Any]] = None, 
    target_jersey: Optional[str] = None
):
    """Process video using Cerebrus tracking system"""
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writers
    out_full = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_high = cv2.VideoWriter(highlight_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize tracker
    tracker = CerebrusTracker(max_age=int(fps*3))
    start_frame = 0
    
    if mode == 'Manual' and initial_target_data:
        tracker.initialize_target(initial_target_data)
        cap.read()  # Skip first frame
        start_frame = 1

    # Process frames
    for frame_idx in tqdm(range(start_frame, total_frames), desc='Processing Video with Cerebrus'):
        ret, frame = cap.read()
        if not ret: 
            break

        # Load detections from JSON (assuming they exist)
        # In a real implementation, you'd integrate this with the detection system
        # For now, we'll create placeholder data
        p_data = []  # This would come from the detection system
        b_detections = np.array([])  # This would come from ball detection
        
        # Update tracker
        auto_target = target_jersey if mode == 'Automatic' else None
        if mode == 'Automatic' and target_jersey == '': 
            auto_target = ''
        
        tracked_players, ball_pos = tracker.update(frame, p_data, b_detections, target_jersey_number=auto_target)

        # Process tracking results
        is_highlight_frame = False
        if tracked_players:
            track = tracked_players[0]
            x1, y1, x2, y2 = map(int, track.bbox)
            label = f"PLAYER {track.confirmed_number}" if track.confirmed_number else "TARGET PLAYER"
            state_label = f"STATE: {track.state.upper()}"
            
            # Color coding based on state
            color = (0, 255, 0)  # Green for confirmed
            if track.state == 'lost': 
                color = (0, 165, 255)  # Orange for lost
            elif track.state == 'tentative': 
                color = (255, 255, 0)  # Yellow for tentative
            
            # Draw bounding box and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, state_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            is_highlight_frame = True

        # Draw ball if detected
        if ball_pos:
            cv2.circle(frame, ball_pos, 15, (0, 255, 255), -1)
            cv2.circle(frame, ball_pos, 15, (0,0,0), 2)
            
            # Highlight frame if player is near ball
            if tracked_players:
                player_center = ((x1+x2)//2, (y1+y2)//2)
                if np.linalg.norm(np.array(player_center) - np.array(ball_pos)) < 250: 
                    is_highlight_frame = True

        # Write to appropriate output
        if is_highlight_frame:
            out_high.write(frame)
        
        out_full.write(frame)

    # Cleanup
    cap.release()
    out_full.release()
    out_high.release()
    
    print("\n--- Processing Complete ---")
    print(f"Full video saved to: {output_path}")
    print(f"Highlights video saved to: {highlight_path}")

if __name__ == "__main__":
    print("Cerebrus Tracking System v9 - Continuity-Guided Re-ID")
    print("This module provides enhanced tracking capabilities for the main pipeline")
