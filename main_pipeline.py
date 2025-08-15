import os
import sys
import argparse
from player_detection import SoccerPlayerDetector
from bytetrack_tracker import process_tracking
from reid_system import PlayerReID
from event_detection import EventDetector
from video_assembly import VideoAssembler

class SoccerHighlightPipeline:
    def __init__(self):
        self.detector = SoccerPlayerDetector()
        self.reid = PlayerReID()
        self.event_detector = EventDetector()
        self.assembler = VideoAssembler()

    def run_full_pipeline(self, video_path, target_player_id, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== SOCCER PLAYER HIGHLIGHT PIPELINE ===")
        print(f"Processing video: {video_path}")
        print(f"Target player ID: {target_player_id}")
        print(f"Output directory: {output_dir}")
        print()

        print("Stage 1: Player Detection")
        detections_path = os.path.join(output_dir, "detections.json")
        self.detector.process_video(video_path, detections_path)
        print("✓ Player detection completed")
        print()

        print("Stage 2: Short-term Tracking")
        tracklets_path = os.path.join(output_dir, "tracklets.json")
        process_tracking(detections_path, tracklets_path)
        print("✓ Short-term tracking completed")
        print()

        print("Stage 3: Long-term Re-Identification")
        long_tracks_path = os.path.join(output_dir, "long_player_track.json")
        self.reid.process_tracklets(tracklets_path, video_path, long_tracks_path)
        print("✓ Long-term Re-ID completed")
        print()

        print("Stage 4: Event Detection")
        events_path = os.path.join(output_dir, "events.json")
        self.event_detector.detect_events(video_path, long_tracks_path, events_path)
        
        player_events_path = os.path.join(output_dir, "player_events.json")
        self.event_detector.filter_player_events(events_path, long_tracks_path, target_player_id, player_events_path)
        print("✓ Event detection completed")
        print()

        print("Stage 5: Video Assembly")
        highlight_path = os.path.join(output_dir, f"player_{target_player_id}_highlights.mp4")
        summary_path = os.path.join(output_dir, f"player_{target_player_id}_summary.mp4")
        
        self.assembler.assemble_highlight_reel(video_path, player_events_path, long_tracks_path, target_player_id, highlight_path)
        self.assembler.create_player_summary(video_path, long_tracks_path, target_player_id, summary_path)
        print("✓ Video assembly completed")
        print()

        print("=== PIPELINE COMPLETED ===")
        print(f"Highlight reel: {highlight_path}")
        print(f"Player summary: {summary_path}")
        
        self.assembler.cleanup()

    def run_detection_only(self, video_path, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running player detection only...")
        detections_path = os.path.join(output_dir, "detections.json")
        self.detector.process_video(video_path, detections_path)
        print(f"Detections saved to: {detections_path}")

    def run_tracking_only(self, detections_path, output_dir="output"):
        print("Running tracking only...")
        tracklets_path = os.path.join(output_dir, "tracklets.json")
        process_tracking(detections_path, tracklets_path)
        print(f"Tracklets saved to: {tracklets_path}")

def main():
    parser = argparse.ArgumentParser(description="Soccer Player Highlight Pipeline")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--player-id", type=int, default=1, help="Target player ID (default: 1)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--stage", choices=["full", "detection", "tracking"], default="full", 
                       help="Pipeline stage to run (default: full)")
    parser.add_argument("--detections", help="Path to detections.json (for tracking stage)")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        sys.exit(1)

    pipeline = SoccerHighlightPipeline()

    if args.stage == "full":
        pipeline.run_full_pipeline(args.video_path, args.player_id, args.output_dir)
    elif args.stage == "detection":
        pipeline.run_detection_only(args.video_path, args.output_dir)
    elif args.stage == "tracking":
        if not args.detections:
            print("Error: --detections path required for tracking stage")
            sys.exit(1)
        pipeline.run_tracking_only(args.detections, args.output_dir)

if __name__ == "__main__":
    main()
