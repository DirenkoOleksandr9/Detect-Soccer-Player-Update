import cv2
import json
import subprocess
import os
import tempfile
import shutil
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class VideoAssembler:
    def __init__(self):
        self.clip_duration = 5.0
        self.max_highlight_duration = 300.0
        self.temp_dir = tempfile.mkdtemp()

    def find_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]

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

    def assemble_highlight_reel(self, video_path: str, player_events_path: str, output_path: str, target_player_id: int) -> bool:
        with open(player_events_path, 'r') as f:
            all_events = json.load(f)
        
        player_events = [e for e in all_events if e.get('player_id') == target_player_id or 'cluster' in e.get('event_type', '') or 'goal' in e.get('event_type', '')]
        
        if not player_events:
            print("No events for target player. Cannot create reel.")
            return False
        
        print("Finding scene cuts... (this may take a moment)")
        scenes = self.find_scenes(video_path)
        event_timestamps = sorted([e['timestamp'] for e in player_events])
        
        clips_to_extract = []
        total_duration = 0.0
        for ts in event_timestamps:
            if total_duration >= self.max_highlight_duration: break
            for start, end in scenes:
                if start <= ts <= end:
                    clip_duration = end - start
                    if total_duration + clip_duration <= self.max_highlight_duration:
                        if not any(abs(c['start'] - start) < 1.0 for c in clips_to_extract):
                            clips_to_extract.append({'start': start, 'end': end})
                            total_duration += clip_duration
                    break
        
        if not clips_to_extract:
            print("Could not map any events to scenes.")
            return False
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        clip_paths = []
        for i, clip in enumerate(tqdm(clips_to_extract, desc="Extracting scene clips")):
            clip_path = os.path.join(self.temp_dir, f"clip_{i:04d}.mp4")
            command = ['ffmpeg', '-y', '-i', video_path, '-ss', str(clip['start']), '-to', str(clip['end']), '-c', 'copy', '-avoid_negative_ts', '1', clip_path]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0: clip_paths.append(clip_path)
        
        concat_list_path = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for path in clip_paths:
                f.write(f"file '{os.path.basename(path)}'\n")
        
        concat_command = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', output_path]
        concat_result = subprocess.run(concat_command, cwd=self.temp_dir, capture_output=True, text=True)

        if concat_result.returncode == 0:
            print(f"Highlight reel created: {output_path}")
            print(f"Total duration: {total_duration:.2f} seconds")
            return True
        else:
            print("Failed to concatenate clips. FFmpeg error:", concat_result.stderr)
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
