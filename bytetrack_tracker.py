import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import json
from tqdm import tqdm

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(7, 4)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, 0, 0],
                                                 [0, 0, 1, 0, 0, 0, 0],
                                                 [0, 0, 0, 1, 0, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                                [0, 1, 0, 0, 0, 1, 0],
                                                [0, 0, 1, 0, 0, 0, 1],
                                                [0, 0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 0, 1]], np.float32) * 0.03

    def predict(self):
        return self.kalman.predict()

    def update(self, measurement):
        return self.kalman.correct(measurement)

class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.kalman = KalmanFilter()
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        self.kalman.kalman.statePre = np.array([[center_x], [center_y], [width], [height], [0], [0], [0]], np.float32)
        self.kalman.kalman.statePost = np.array([[center_x], [center_y], [width], [height], [0], [0], [0]], np.float32)

    def predict(self):
        prediction = self.kalman.predict()
        center_x, center_y, width, height = prediction[:4].flatten()
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.hits += 1
        self.time_since_update = 0
        
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        measurement = np.array([[center_x], [center_y], [width], [height]], np.float32)
        self.kalman.update(measurement)

class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_tracks = []
        self.lost_tracks = []
        self.frame_id = 0
        self.next_id = 1

    def iou_distance(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.array([])
        
        track_boxes = np.array([track.bbox for track in tracks])
        detection_boxes = np.array([det['bbox'] for det in detections])
        
        def box_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = box_iou(track.bbox, detection['bbox'])
        
        return 1 - iou_matrix

    def update(self, detections):
        self.frame_id += 1
        
        activated_tracks = []
        refined_tracks = []
        lost_tracks = []
        
        for track in self.tracked_tracks:
            track.predict()
        
        track_pool = self.tracked_tracks + self.lost_tracks
        
        if len(track_pool) > 0:
            iou_dists = self.iou_distance(track_pool, detections)
            if len(iou_dists) > 0:
                track_indices, detection_indices = linear_sum_assignment(iou_dists)
                for track_idx, detection_idx in zip(track_indices, detection_indices):
                    if iou_dists[track_idx, detection_idx] <= 1 - self.match_thresh:
                        track = track_pool[track_idx]
                        track.update(detections[detection_idx])
                        activated_tracks.append(track)
                        if track in self.lost_tracks:
                            self.lost_tracks.remove(track)
                        refined_tracks.append(track)
        
        for track in self.tracked_tracks:
            if track not in refined_tracks:
                track.time_since_update += 1
                if track.time_since_update > self.track_buffer:
                    self.lost_tracks.append(track)
                else:
                    lost_tracks.append(track)
        
        for detection in detections:
            if detection['confidence'] > self.track_thresh:
                track = Track(detection, self.next_id)
                self.next_id += 1
                activated_tracks.append(track)
        
        self.tracked_tracks = activated_tracks + lost_tracks
        
        return self.tracked_tracks

def process_tracking(detections_path, output_path):
    with open(detections_path, 'r') as f:
        all_detections = json.load(f)
    
    tracker = ByteTracker()
    all_tracklets = []
    
    for frame_data in tqdm(all_detections, desc="Tracking players"):
        detections = frame_data['detections']
        tracks = tracker.update(detections)
        
        frame_tracklets = {
            "frame_id": frame_data['frame_id'],
            "tracks": []
        }
        
        for track in tracks:
            frame_tracklets["tracks"].append({
                "track_id": track.track_id,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "age": track.age,
                "hits": track.hits
            })
        
        all_tracklets.append(frame_tracklets)
    
    with open(output_path, 'w') as f:
        json.dump(all_tracklets, f)
    
    return all_tracklets

if __name__ == "__main__":
    process_tracking("detections.json", "tracklets.json")
