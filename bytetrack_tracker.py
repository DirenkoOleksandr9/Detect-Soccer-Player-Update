import cv2
import numpy as np
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import torch

class STrack:
    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.track_id = 0
        self.state = 'new'
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        
        self.kalman_filter = self.init_kalman_filter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
    def init_kalman_filter(self):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        kf.R[2:, 2:] *= 10.
        kf.P[4:, 4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        return kf
    
    def tlwh_to_xyah(self, tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        ret[:2] = ret[:2] + ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def xyah_to_tlwh(self, xyah):
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self._tlwh = self.xyah_to_tlwh(self.mean[:4])
    
    def update(self, detection):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(detection['bbox'])
        )
        self._tlwh = self.xyah_to_tlwh(self.mean[:4])
        self.score = detection['confidence']
    
    def activate(self, frame_id, track_id):
        self.track_id = track_id
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = 'tracked'
        self.is_activated = True
    
    def re_activate(self, new_track, frame_id):
        self._tlwh = new_track['bbox']
        self.score = new_track['confidence']
        self.update(new_track)
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
    
    @property
    def tlwh(self):
        return self._tlwh.copy()
    
    @property
    def tlbr(self):
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

def iou_distance(atracks, btracks):
    if not atracks or not btracks:
        return np.empty((0, 0))
    
    atlbrs = np.array([track.tlbr for track in atracks])
    btlbrs = np.array([track.tlbr for track in btracks])
    
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    
    for i, a in enumerate(atlbrs):
        for j, b in enumerate(btlbrs):
            box_inter = [
                max(a[0], b[0]), max(a[1], b[1]),
                min(a[2], b[2]), min(a[3], b[3])
            ]
            inter_area = max(0, box_inter[2] - box_inter[0]) * max(0, box_inter[3] - box_inter[1])
            union_area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter_area
            
            if union_area > 0:
                ious[i, j] = inter_area / union_area
    
    return 1 - ious

class ByteTracker:
    def __init__(self, high_thresh=0.6, low_thresh=0.1, max_time_lost=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_id_count = 0
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        
        print(f"ByteTracker initialized: high_thresh={high_thresh}, low_thresh={low_thresh}, max_time_lost={max_time_lost}")
    
    def update(self, detections):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        dets_high = [d for d in detections if d['confidence'] >= self.high_thresh]
        dets_low = [d for d in detections if self.low_thresh <= d['confidence'] < self.high_thresh]
        
        stracks_high = []
        for d in dets_high:
            x1, y1, x2, y2 = d['bbox']
            stracks_high.append(STrack([x1, y1, x2 - x1, y2 - y1], d['confidence']))
        
        stracks_low = []
        for d in dets_low:
            x1, y1, x2, y2 = d['bbox']
            stracks_low.append(STrack([x1, y1, x2 - x1, y2 - y1], d['confidence']))
        
        for strack in self.tracked_stracks:
            strack.predict()
        
        dists = iou_distance(self.tracked_stracks, stracks_high)
        matches, u_track, u_detection = self.linear_assignment(dists, 0.8)
        
        for i, j in matches:
            track = self.tracked_stracks[i]
            det = stracks_high[j]
            track.update(det)
            activated_starcks.append(track)
        
        unmatched_tracks = [self.tracked_stracks[i] for i in u_track]
        dists = iou_distance(unmatched_tracks, stracks_low)
        matches, u_track, u_detection_low = self.linear_assignment(dists, 0.5)
        
        for i, j in matches:
            track = unmatched_tracks[i]
            det = stracks_low[j]
            track.update(det)
            activated_starcks.append(track)
        
        for i in u_track:
            track = unmatched_tracks[i]
            track.state = 'lost'
            lost_stracks.append(track)
        
        for i in u_detection:
            track = stracks_high[i]
            if track.score >= self.high_thresh:
                self.track_id_count += 1
                track.activate(self.frame_id, self.track_id_count)
                activated_starcks.append(track)
        
        for track in self.lost_stracks:
            track.time_since_update += 1
            if track.time_since_update > self.max_time_lost:
                track.state = 'removed'
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked'] + activated_starcks
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 'lost'] + lost_stracks
        self.removed_stracks = [t for t in self.removed_stracks if t.state == 'removed'] + removed_stracks
        
        output = []
        for t in self.tracked_stracks:
            if t.is_activated:
                output.append({
                    'track_id': t.track_id,
                    'bbox': [int(x) for x in t.tlbr],
                    'confidence': t.score,
                    'state': t.state
                })
        
        return output
    
    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        rows, cols = linear_sum_assignment(cost_matrix)
        matches = [(r, c) for r, c in zip(rows, cols) if cost_matrix[r, c] < thresh]
        u_track = [r for r in range(cost_matrix.shape[0]) if r not in [m[0] for m in matches]]
        u_det = [c for c in range(cost_matrix.shape[1]) if c not in [m[1] for m in matches]]
        
        return matches, u_track, u_det

def process_tracking(detections_path, output_path):
    with open(detections_path, 'r') as f:
        all_detections = json.load(f)
    
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, max_time_lost=30)
    all_tracklets = []
    
    with tqdm(total=len(all_detections), desc="Stage 2: Advanced Tracking") as pbar:
        for frame_data in all_detections:
            frame_id = frame_data['frame_id']
            detections = frame_data['detections']['players']
            
            tracks = tracker.update(detections)
            
            frame_tracklets = {
                "frame_id": frame_id,
                "tracks": tracks
            }
            
            all_tracklets.append(frame_tracklets)
            pbar.update(1)
    
    with open(output_path, 'w') as f:
        json.dump(all_tracklets, f, indent=2)
    
    print(f"Tracking complete. Saved to {output_path}")
    print(f"Total tracklets: {len(all_tracklets)} frames")
    
    return all_tracklets

if __name__ == "__main__":
    process_tracking("detections.json", "tracklets.json")
