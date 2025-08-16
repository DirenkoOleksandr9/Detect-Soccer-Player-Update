import os
import argparse
import json
import cv2
from bytetrack_tracker import process_tracking
from reid_system import PlayerReID
from advanced_event_detection import AdvancedEventDetector
from video_assembly import VideoAssembler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--player-id", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tracklets_path = os.path.join(args.output_dir, "tracklets.json")
    long_tracks_path = os.path.join(args.output_dir, "long_player_track.json")
    events_path = os.path.join(args.output_dir, "events.json")
    player_events_path = os.path.join(args.output_dir, "player_events.json")
    highlight_path = os.path.join(args.output_dir, f"player_{args.player_id}_highlights.mp4")

    process_tracking(args.detections, tracklets_path)

    reid = PlayerReID()
    reid.process_tracklets(tracklets_path, args.video, long_tracks_path)

    cap = cv2.VideoCapture(args.video)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    detector = AdvancedEventDetector(video_width, video_height, fps)
    detector.detect_events(long_tracks_path, events_path)
    detector.filter_player_events(events_path, args.player_id, player_events_path)

    assembler = VideoAssembler()
    assembler.assemble_highlight_reel(args.video, player_events_path, highlight_path, args.player_id)
    assembler.cleanup()

if __name__ == "__main__":
    main()


