import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import cv2

class TrackletStitcher:
    def __init__(self, max_gap_frames=300, similarity_threshold=0.7):
        self.max_gap_frames = max_gap_frames
        self.similarity_threshold = similarity_threshold
        print(f"TrackletStitcher initialized: max_gap={max_gap_frames}, sim_thresh={similarity_threshold}")
    
    def extract_tracklet_features(self, tracklet_data, video_path):
        """Extract appearance features from tracklet frames"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        features = []
        
        for frame_data in tracklet_data:
            frame_id = frame_data['frame_id']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Use the first player detection in the frame
            if frame_data['players']:
                player = frame_data['players'][0]
                x1, y1, x2, y2 = player['bbox']
                
                # Clamp coordinates
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                if x2 > x1 and y2 > y1:
                    patch = frame[y1:y2, x1:x2]
                    if patch.size > 0:
                        # Simple color histogram as feature
                        hsv = cv2.cvtColor(cv2.resize(patch, (64, 64)), cv2.COLOR_BGR2HSV)
                        hist = cv2.normalize(
                            cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]), 
                            None
                        ).flatten()
                        features.append(hist)
        
        cap.release()
        
        if features:
            return np.mean(features, axis=0)
        return None
    
    def calculate_tracklet_similarity(self, tracklet1, tracklet2, video_path):
        """Calculate similarity between two tracklets"""
        feat1 = self.extract_tracklet_features(tracklet1, video_path)
        feat2 = self.extract_tracklet_features(tracklet2, video_path)
        
        if feat1 is None or feat2 is None:
            return 0.0
        
        return cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0][0]
    
    def find_stitchable_tracklets(self, long_tracks, video_path):
        """Find tracklets that can be stitched together"""
        print("Analyzing tracklets for stitching opportunities...")
        
        # Group tracklets by permanent_id
        tracklets_by_id = {}
        for frame_data in long_tracks:
            for player in frame_data['players']:
                pid = player['permanent_id']
                if pid not in tracklets_by_id:
                    tracklets_by_id[pid] = []
                tracklets_by_id[pid].append(frame_data)
        
        # Sort tracklets by frame_id for each player
        for pid in tracklets_by_id:
            tracklets_by_id[pid].sort(key=lambda x: x['frame_id'])
        
        stitch_candidates = []
        
        # Find gaps in tracking for each player
        for pid, tracklets in tracklets_by_id.items():
            if len(tracklets) < 2:
                continue
            
            for i in range(len(tracklets) - 1):
                gap = tracklets[i + 1]['frame_id'] - tracklets[i]['frame_id']
                if gap > 10 and gap <= self.max_gap_frames:
                    stitch_candidates.append({
                        'player_id': pid,
                        'tracklet1': tracklets[i],
                        'tracklet2': tracklets[i + 1],
                        'gap_frames': gap
                    })
        
        print(f"Found {len(stitch_candidates)} potential stitch candidates")
        return stitch_candidates
    
    def stitch_tracklets(self, long_tracks, video_path):
        """Main stitching function"""
        stitch_candidates = self.find_stitchable_tracklets(long_tracks, video_path)
        
        if not stitch_candidates:
            print("No stitchable tracklets found")
            return long_tracks
        
        # Sort by gap size (smaller gaps first)
        stitch_candidates.sort(key=lambda x: x['gap_frames'])
        
        stitched_count = 0
        
        for candidate in tqdm(stitch_candidates, desc="Stitching tracklets"):
            tracklet1 = candidate['tracklet1']
            tracklet2 = candidate['tracklet2']
            
            # Calculate similarity
            similarity = self.calculate_tracklet_similarity([tracklet1], [tracklet2], video_path)
            
            if similarity >= self.similarity_threshold:
                # Stitch the tracklets by filling the gap
                gap_start = tracklet1['frame_id']
                gap_end = tracklet2['frame_id']
                
                # Find the player in both tracklets
                player1 = None
                player2 = None
                
                for player in tracklet1['players']:
                    if player['permanent_id'] == candidate['player_id']:
                        player1 = player
                        break
                
                for player in tracklet2['players']:
                    if player['permanent_id'] == candidate['player_id']:
                        player2 = player
                        break
                
                if player1 and player2:
                    # Interpolate positions for gap frames
                    for frame_id in range(gap_start + 1, gap_end):
                        # Linear interpolation of bounding box
                        alpha = (frame_id - gap_start) / (gap_end - gap_start)
                        bbox1 = np.array(player1['bbox'])
                        bbox2 = np.array(player2['bbox'])
                        interpolated_bbox = bbox1 * (1 - alpha) + bbox2 * alpha
                        
                        # Create interpolated frame data
                        interpolated_frame = {
                            "frame_id": frame_id,
                            "players": [{
                                "permanent_id": candidate['player_id'],
                                "bbox": interpolated_bbox.tolist(),
                                "jersey": player1.get('jersey'),
                                "confidence": 0.5,  # Lower confidence for interpolated
                                "interpolated": True
                            }]
                        }
                        
                        # Insert into long_tracks
                        long_tracks.append(interpolated_frame)
                    
                    stitched_count += 1
        
        # Sort by frame_id
        long_tracks.sort(key=lambda x: x['frame_id'])
        
        print(f"Stitched {stitched_count} tracklet gaps")
        return long_tracks

def stitch_tracklets_offline(long_tracks_path, video_path, output_path):
    """Offline tracklet stitching function"""
    with open(long_tracks_path, 'r') as f:
        long_tracks = json.load(f)
    
    stitcher = TrackletStitcher(max_gap_frames=300, similarity_threshold=0.7)
    stitched_tracks = stitcher.stitch_tracklets(long_tracks, video_path)
    
    with open(output_path, 'w') as f:
        json.dump(stitched_tracks, f, indent=2)
    
    print(f"Stitched tracklets saved to {output_path}")
    return stitched_tracks

if __name__ == "__main__":
    stitch_tracklets_offline("long_player_track.json", "9.mp4", "stitched_tracklets.json")
