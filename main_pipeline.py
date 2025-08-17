import argparse
import os
import json
from player_detection import SoccerPlayerDetector
from bytetrack_tracker import process_tracking
from reid_system import AdvancedPlayerReID
from motion_compensation import apply_gmc_to_pipeline
from stitch_tracklets import stitch_tracklets_offline
from event_detection import AdvancedEventDetector
from video_assembly import VideoAssembler

def run_full_pipeline(video_path, output_dir, target_jersey=None):
    """Run the complete advanced pipeline with all improvements"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("⚽ ADVANCED SOCCER PLAYER TRACKING PIPELINE v3.0")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Target Jersey: {target_jersey}")
    print("=" * 60)
    
    # Stage 1: SOTA Detection
    print("\n🚀 Stage 1: State-of-the-Art Detection")
    detector = SoccerPlayerDetector(model_name='yolov8x.pt', conf_thresh=0.3, min_area=500)
    detections_path = os.path.join(output_dir, "detections.json")
    detector.process_video(video_path, detections_path)
    
    # Stage 2: Advanced Tracking with Kalman Filter
    print("\n🎯 Stage 2: Advanced Predictive Tracking")
    tracklets_path = os.path.join(output_dir, "tracklets.json")
    process_tracking(detections_path, tracklets_path)
    
    # Stage 3: Global Motion Compensation
    print("\n📐 Stage 3: Global Motion Compensation")
    gmc_tracklets_path = apply_gmc_to_pipeline(video_path, detections_path, tracklets_path, output_dir)
    
    # Stage 4: Advanced Re-Identification
    print("\n🔄 Stage 4: Advanced Re-Identification")
    reid = AdvancedPlayerReID(similarity_threshold=0.6, jersey_bonus=0.3)
    long_tracks_path = os.path.join(output_dir, "long_player_track.json")
    reid.process_tracklets(gmc_tracklets_path, video_path, long_tracks_path)
    
    # Stage 5: Post-Processing - Stitch Tracklets
    print("\n🔗 Stage 5: Post-Processing - Stitch Tracklets")
    stitched_tracks_path = os.path.join(output_dir, "stitched_tracklets.json")
    stitch_tracklets_offline(long_tracks_path, video_path, stitched_tracks_path)
    
    # Stage 6: Event Detection
    print("\n⚡ Stage 6: Advanced Event Detection")
    event_detector = AdvancedEventDetector()
    events_path = os.path.join(output_dir, "player_events.json")
    event_detector.detect_events(stitched_tracks_path, video_path, events_path, target_jersey)
    
    # Stage 7: Video Assembly
    print("\n🎬 Stage 7: Professional Video Assembly")
    assembler = VideoAssembler()
    highlight_path = os.path.join(output_dir, "player_highlight_reel.mp4")
    assembler.assemble_highlight_reel(
        stitched_tracks_path, 
        events_path, 
        video_path, 
        highlight_path, 
        target_jersey
    )
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    
    # Generate summary
    generate_pipeline_summary(output_dir, target_jersey)
    
    return {
        'detections': detections_path,
        'tracklets': tracklets_path,
        'gmc_tracklets': gmc_tracklets_path,
        'long_tracks': long_tracks_path,
        'stitched_tracks': stitched_tracks_path,
        'events': events_path,
        'highlight': highlight_path
    }

def run_detection_only(video_path, output_dir):
    """Run only the detection stage"""
    print("🔍 Running Detection Only")
    detector = SoccerPlayerDetector(model_name='yolov8x.pt', conf_thresh=0.3, min_area=500)
    detections_path = os.path.join(output_dir, "detections.json")
    detector.process_video(video_path, detections_path)
    return detections_path

def run_tracking_only(detections_path, output_dir):
    """Run only the tracking stage"""
    print("🎯 Running Tracking Only")
    tracklets_path = os.path.join(output_dir, "tracklets.json")
    process_tracking(detections_path, tracklets_path)
    return tracklets_path

def generate_pipeline_summary(output_dir, target_jersey):
    """Generate a comprehensive summary of the pipeline results"""
    summary = {
        'pipeline_version': '3.0',
        'target_jersey': target_jersey,
        'output_files': {}
    }
    
    # Check which files were generated
    expected_files = [
        'detections.json',
        'tracklets.json', 
        'motion_compensated_tracklets.json',
        'long_player_track.json',
        'stitched_tracklets.json',
        'player_events.json',
        'player_highlight_reel.mp4'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            summary['output_files'][filename] = {
                'exists': True,
                'size_bytes': size,
                'size_mb': round(size / (1024 * 1024), 2)
            }
        else:
            summary['output_files'][filename] = {'exists': False}
    
    # Analyze results if available
    try:
        if os.path.exists(os.path.join(output_dir, 'long_player_track.json')):
            with open(os.path.join(output_dir, 'long_player_track.json'), 'r') as f:
                long_tracks = json.load(f)
            
            unique_players = set()
            total_detections = 0
            confirmed_jerseys = set()
            
            for frame in long_tracks:
                for player in frame['players']:
                    unique_players.add(player['permanent_id'])
                    total_detections += 1
                    if player.get('jersey'):
                        confirmed_jerseys.add(player['jersey'])
            
            summary['statistics'] = {
                'unique_players': len(unique_players),
                'total_detections': total_detections,
                'confirmed_jerseys': len(confirmed_jerseys),
                'jersey_numbers': list(confirmed_jerseys)
            }
        
        if os.path.exists(os.path.join(output_dir, 'player_events.json')):
            with open(os.path.join(output_dir, 'player_events.json'), 'r') as f:
                events = json.load(f)
            
            summary['events'] = {
                'total_events': len(events),
                'event_types': {}
            }
            
            for event in events:
                event_type = event.get('type', 'unknown')
                summary['events']['event_types'][event_type] = summary['events']['event_types'].get(event_type, 0) + 1
    
    except Exception as e:
        summary['analysis_error'] = str(e)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Pipeline Summary saved to: {summary_path}")
    
    # Print key statistics
    if 'statistics' in summary:
        stats = summary['statistics']
        print(f"📈 Results:")
        print(f"   • Unique players tracked: {stats['unique_players']}")
        print(f"   • Total detections: {stats['total_detections']}")
        print(f"   • Confirmed jerseys: {stats['confirmed_jerseys']}")
        if stats['jersey_numbers']:
            print(f"   • Jersey numbers: {sorted(stats['jersey_numbers'])}")
    
    if 'events' in summary:
        events = summary['events']
        print(f"   • Total events detected: {events['total_events']}")
        for event_type, count in events['event_types'].items():
            print(f"   • {event_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Soccer Player Tracking Pipeline v3.0")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--player-id", type=int, help="Target jersey number for highlight reel")
    parser.add_argument("--stage", choices=["full", "detection", "tracking"], default="full",
                       help="Pipeline stage to run")
    parser.add_argument("--detections", help="Path to existing detections.json (for tracking stage)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.stage == "full":
        run_full_pipeline(args.video_path, args.output_dir, args.player_id)
    elif args.stage == "detection":
        run_detection_only(args.video_path, args.output_dir)
    elif args.stage == "tracking":
        if not args.detections:
            print("❌ Error: --detections required for tracking stage")
            return
        run_tracking_only(args.detections, args.output_dir)

if __name__ == "__main__":
    main()
