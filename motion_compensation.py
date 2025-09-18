import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import json

class GlobalMotionCompensation:
    def __init__(self, feature_detector='ORB', max_features=2000):
        self.feature_detector = feature_detector
        self.max_features = max_features
        
        if feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
        
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING if feature_detector == 'ORB' else cv2.NORM_L2)
        
        print(f"GMC initialized with {feature_detector} detector")
    
    def detect_features(self, frame):
        """Detect features in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def estimate_motion(self, kp1, des1, kp2, des2, frame_shape):
        """Estimate motion between two frames"""
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return np.eye(3, dtype=np.float32)
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return np.eye(3, dtype=np.float32)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return np.eye(3, dtype=np.float32)
        
        return H
    
    def compensate_motion(self, bbox, H):
        """Apply motion compensation to bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Convert to homogeneous coordinates
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ], dtype=np.float32)
        
        # Apply transformation
        transformed_corners = (H @ corners.T).T
        
        # Convert back to inhomogeneous coordinates
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
        
        # Get bounding box of transformed corners
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        
        new_x1 = int(np.min(x_coords))
        new_y1 = int(np.min(y_coords))
        new_x2 = int(np.max(x_coords))
        new_y2 = int(np.max(y_coords))
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def process_video_motion(self, video_path, detections_path, output_path):
        """Process video to estimate and compensate for global motion"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with open(detections_path, 'r') as f:
            all_detections = json.load(f)
        
        compensated_detections = []
        motion_matrices = []
        
        prev_frame = None
        prev_kp = None
        prev_des = None
        
        print(f"Processing motion compensation for {total_frames} frames")
        
        with tqdm(total=total_frames, desc="Motion Compensation") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect features in current frame
                kp, des = self.detect_features(frame)
                
                # Estimate motion if we have a previous frame
                if prev_frame is not None and prev_kp is not None and prev_des is not None:
                    H = self.estimate_motion(prev_kp, prev_des, kp, des, frame.shape)
                    motion_matrices.append(H.tolist())
                else:
                    H = np.eye(3, dtype=np.float32)
                    motion_matrices.append(H.tolist())
                
                # Find corresponding detection data
                frame_detection = None
                for det in all_detections:
                    if det['frame_id'] == frame_idx:
                        frame_detection = det
                        break
                
                if frame_detection:
                    compensated_frame = {
                        'frame_id': frame_idx,
                        'detections': {
                            'players': [],
                            'ball': frame_detection['detections'].get('ball')
                        },
                        'motion_matrix': H.tolist()
                    }
                    
                    # Compensate player detections
                    for player in frame_detection['detections']['players']:
                        compensated_bbox = self.compensate_motion(player['bbox'], H)
                        compensated_player = player.copy()
                        compensated_player['bbox'] = compensated_bbox
                        compensated_frame['detections']['players'].append(compensated_player)
                    
                    compensated_detections.append(compensated_frame)
                
                # Update previous frame
                prev_frame = frame.copy()
                prev_kp = kp
                prev_des = des
                
                pbar.update(1)
        
        cap.release()
        
        with open(output_path, 'w') as f:
            json.dump(compensated_detections, f, indent=2)
        
        print(f"Motion compensation complete. Saved to {output_path}")
        return compensated_detections
    
    def apply_motion_to_tracklets(self, tracklets_path, motion_path, output_path):
        """Apply motion compensation to tracklets"""
        with open(tracklets_path, 'r') as f:
            tracklets = json.load(f)
        
        with open(motion_path, 'r') as f:
            motion_data = json.load(f)
        
        # Create motion lookup
        motion_lookup = {frame['frame_id']: np.array(frame['motion_matrix']) for frame in motion_data}
        
        compensated_tracklets = []
        
        for frame_data in tqdm(tracklets, desc="Applying motion to tracklets"):
            frame_id = frame_data['frame_id']
            H = motion_lookup.get(frame_id, np.eye(3))
            
            compensated_frame = {
                'frame_id': frame_id,
                'tracks': []
            }
            
            for track in frame_data['tracks']:
                compensated_bbox = self.compensate_motion(track['bbox'], H)
                compensated_track = track.copy()
                compensated_track['bbox'] = compensated_bbox
                compensated_frame['tracks'].append(compensated_track)
            
            compensated_tracklets.append(compensated_frame)
        
        with open(output_path, 'w') as f:
            json.dump(compensated_tracklets, f, indent=2)
        
        print(f"Motion-compensated tracklets saved to {output_path}")
        return compensated_tracklets

def apply_gmc_to_pipeline(video_path, detections_path, tracklets_path, output_dir):
    """Apply GMC to the entire pipeline"""
    gmc = GlobalMotionCompensation()
    
    # Step 1: Process video motion
    motion_compensated_detections = gmc.process_video_motion(
        video_path, 
        detections_path, 
        f"{output_dir}/motion_compensated_detections.json"
    )
    
    # Step 2: Apply motion to tracklets
    gmc.apply_motion_to_tracklets(
        tracklets_path,
        f"{output_dir}/motion_compensated_detections.json",
        f"{output_dir}/motion_compensated_tracklets.json"
    )
    
    print("GMC pipeline complete")
    return f"{output_dir}/motion_compensated_tracklets.json"

if __name__ == "__main__":
    apply_gmc_to_pipeline("9.mp4", "detections.json", "tracklets.json", "output")
