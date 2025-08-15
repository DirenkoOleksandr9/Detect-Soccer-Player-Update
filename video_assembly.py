import cv2
import json
import subprocess
import os
import tempfile
from tqdm import tqdm
import numpy as np

class VideoAssembler:
    def __init__(self):
        self.clip_duration = 5.0
        self.max_highlight_duration = 300.0
        self.temp_dir = tempfile.mkdtemp()

    def find_scene_boundaries(self, video_path, event_timestamps):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scene_boundaries = []
        
        for timestamp in event_timestamps:
            frame_idx = int(timestamp * fps)
            
            start_frame = max(0, frame_idx - int(self.clip_duration * fps // 2))
            end_frame = min(total_frames, frame_idx + int(self.clip_duration * fps // 2))
            
            scene_boundaries.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame / fps,
                'end_time': end_frame / fps,
                'event_timestamp': timestamp
            })
        
        cap.release()
        return scene_boundaries

    def extract_clip(self, video_path, start_time, end_time, output_path):
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def create_concat_list(self, clip_paths, output_path):
        with open(output_path, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")

    def concatenate_clips(self, concat_list_path, output_path):
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def assemble_highlight_reel(self, video_path, player_events_path, player_tracks_path, target_player_id, output_path):
        with open(player_events_path, 'r') as f:
            player_events = json.load(f)
        
        with open(player_tracks_path, 'r') as f:
            player_tracks = json.load(f)
        
        event_timestamps = [event['timestamp'] for event in player_events]
        
        if not event_timestamps:
            print(f"No events found for player {target_player_id}")
            return False
        
        scene_boundaries = self.find_scene_boundaries(video_path, event_timestamps)
        
        clip_paths = []
        total_duration = 0.0
        
        for i, boundary in enumerate(scene_boundaries):
            if total_duration >= self.max_highlight_duration:
                break
            
            clip_path = os.path.join(self.temp_dir, f"clip_{i:04d}.mp4")
            
            if self.extract_clip(video_path, boundary['start_time'], boundary['end_time'], clip_path):
                clip_paths.append(clip_path)
                total_duration += boundary['end_time'] - boundary['start_time']
        
        if not clip_paths:
            print("No clips were successfully extracted")
            return False
        
        concat_list_path = os.path.join(self.temp_dir, "concat_list.txt")
        self.create_concat_list(clip_paths, concat_list_path)
        
        if self.concatenate_clips(concat_list_path, output_path):
            print(f"Highlight reel created: {output_path}")
            print(f"Total duration: {total_duration:.2f} seconds")
            return True
        else:
            print("Failed to concatenate clips")
            return False

    def create_player_summary(self, video_path, player_tracks_path, target_player_id, output_path):
        with open(player_tracks_path, 'r') as f:
            player_tracks = json.load(f)
        
        player_frames = []
        
        for frame_data in player_tracks:
            for player in frame_data['players']:
                if player['permanent_id'] == target_player_id:
                    player_frames.append({
                        'frame_id': frame_data['frame_id'],
                        'bbox': player['bbox'],
                        'jersey': player.get('jersey')
                    })
                    break
        
        if not player_frames:
            print(f"No frames found for player {target_player_id}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        player_frame_idx = 0
        
        with tqdm(total=len(player_frames), desc="Creating player summary") as pbar:
            while cap.isOpened() and player_frame_idx < len(player_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if player_frame_idx < len(player_frames) and frame_idx == player_frames[player_frame_idx]['frame_id']:
                    bbox = player_frames[player_frame_idx]['bbox']
                    jersey = player_frames[player_frame_idx]['jersey']
                    
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"Player {target_player_id}"
                    if jersey is not None:
                        label += f" #{jersey}"
                    
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    player_frame_idx += 1
                    pbar.update(1)
                
                out.write(frame)
                frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"Player summary created: {output_path}")
        return True

    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

if __name__ == "__main__":
    assembler = VideoAssembler()
    
    target_player_id = 1
    video_path = "9.mp4"
    
    assembler.assemble_highlight_reel(
        video_path,
        "player_events.json",
        "long_player_track.json",
        target_player_id,
        f"player_{target_player_id}_highlights.mp4"
    )
    
    assembler.create_player_summary(
        video_path,
        "long_player_track.json",
        target_player_id,
        f"player_{target_player_id}_summary.mp4"
    )
    
    assembler.cleanup()
