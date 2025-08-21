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
from cerebrus_tracker import CerebrusTracker, process_video_with_cerebrus

def run_full_pipeline(video_path, output_dir, target_jersey=None, tracking_mode='Enhanced'):
    """Run the complete advanced pipeline with all improvements"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("⚽ ADVANCED SOCCER PLAYER TRACKING PIPELINE v4.0")
    print("🎯 Enhanced with Cerebrus Tracking System v9")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Target Jersey: {target_jersey}")
    print(f"Tracking Mode: {tracking_mode}")
    print("=" * 60)
    
    # Stage 1: Enhanced Detection with OpenCLIP & OCR
    print("\n🚀 Stage 1: Enhanced Detection & Feature Extraction")
    print("   - YOLOv8-Large for player detection")
    print("   - YOLOv8n for ball detection") 
    print("   - OpenCLIP ViT-H/14 embeddings")
    print("   - Jersey number OCR with Tesseract")
    print("   - Color histogram analysis")
    print("   - SSIM similarity support")
    
    detector = SoccerPlayerDetector(model_name='yolov8l.pt', conf_thresh=0.4, min_area=500)
    detections_path = os.path.join(output_dir, "enhanced_detections.json")
    detector.process_video(video_path, detections_path)
    
    if tracking_mode == 'Cerebrus':
        # Use the new Cerebrus tracking system
        print("\n🎯 Stage 2: Cerebrus Tracking System v9")
        print("   - Continuity-Guided Re-Identification")
        print("   - Intelligent Fused Score matching")
        print("   - Ball tracking integration")
        print("   - Enhanced state management")
        
        # Process with Cerebrus tracker
        output_video_path = os.path.join(output_dir, "tracked_output_cerebrus_v9.mp4")
        highlight_video_path = os.path.join(output_dir, "highlights_cerebrus_v9.mp4")
        
        process_video_with_cerebrus(
            input_path=video_path,
            output_path=output_video_path,
            highlight_path=highlight_video_path,
            mode='Automatic',
            target_jersey=target_jersey
        )
        
        # Generate summary for Cerebrus mode
        generate_cerebrus_summary(output_dir, target_jersey)
        
        return {
            'detections': detections_path,
            'output_video': output_video_path,
            'highlight_video': highlight_video_path,
            'tracking_mode': 'Cerebrus'
        }
    
    else:
        # Use the traditional pipeline
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
        print("✅ TRADITIONAL PIPELINE COMPLETE!")
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
            'highlight': highlight_path,
            'tracking_mode': 'Traditional'
        }

def run_cerebrus_only(video_path, output_dir, target_jersey=None):
    """Run only the Cerebrus tracking system"""
    print("🎯 Running Cerebrus Tracking System v9 Only")
    
    # First run detection to get enhanced features
    detector = SoccerPlayerDetector(model_name='yolov8l.pt', conf_thresh=0.4, min_area=500)
    detections_path = os.path.join(output_dir, "enhanced_detections.json")
    detector.process_video(video_path, detections_path)
    
    # Then run Cerebrus tracking
    output_video_path = os.path.join(output_dir, "tracked_output_cerebrus_v9.mp4")
    highlight_video_path = os.path.join(output_dir, "highlights_cerebrus_v9.mp4")
    
    process_video_with_cerebrus(
        input_path=video_path,
        output_path=output_video_path,
        highlight_path=highlight_video_path,
        mode='Automatic',
        target_jersey=target_jersey
    )
    
    return {
        'detections': detections_path,
        'output_video': output_video_path,
        'highlight_video': highlight_video_path
    }

def run_detection_only(video_path, output_dir):
    """Run only the detection stage"""
    print("🔍 Running Enhanced Detection Only")
    detector = SoccerPlayerDetector(model_name='yolov8l.pt', conf_thresh=0.4, min_area=500)
    detections_path = os.path.join(output_dir, "enhanced_detections.json")
    detector.process_video(video_path, detections_path)
    return detections_path

def run_tracking_only(detections_path, output_dir):
    """Run only the tracking stage"""
    print("🎯 Running Tracking Only")
    tracklets_path = os.path.join(output_dir, "tracklets.json")
    process_tracking(detections_path, tracklets_path)
    return tracklets_path

def generate_cerebrus_summary(output_dir, target_jersey):
    """Generate summary for Cerebrus tracking mode"""
    print("\n" + "=" * 60)
    print("🎯 CEREBRUS TRACKING SYSTEM v9 SUMMARY")
    print("=" * 60)
    print(f"Output Directory: {output_dir}")
    print(f"Target Jersey: {target_jersey or 'Auto-detected'}")
    print("\nGenerated Files:")
    print(f"  📹 Enhanced Detections: enhanced_detections.json")
    print(f"  🎬 Full Tracked Video: tracked_output_cerebrus_v9.mp4")
    print(f"  ⭐ Highlights Video: highlights_cerebrus_v9.mp4")
    print("\nFeatures:")
    print("  ✅ YOLOv8-Large player detection")
    print("  ✅ YOLOv8n ball detection")
    print("  ✅ OpenCLIP ViT-H/14 embeddings")
    print("  ✅ Jersey number OCR")
    print("  ✅ Color histogram analysis")
    print("  ✅ Continuity-Guided Re-ID")
    print("  ✅ Intelligent Fused Score matching")
    print("  ✅ Enhanced state management")
    print("=" * 60)

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
    parser = argparse.ArgumentParser(description="Advanced Soccer Player Tracking Pipeline v4.0 - Enhanced with Cerebrus Tracking System v9")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--player-id", type=int, help="Target jersey number for highlight reel")
    parser.add_argument("--stage", choices=["full", "detection", "tracking", "cerebrus"], default="full",
                       help="Pipeline stage to run")
    parser.add_argument("--detections", help="Path to existing detections.json (for tracking stage)")
    parser.add_argument("--tracking-mode", choices=["Traditional", "Cerebrus"], default="Cerebrus",
                       help="Tracking mode: Traditional (full pipeline) or Cerebrus (enhanced single-player)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.stage == "full":
        if args.tracking_mode == "Cerebrus":
            print("🎯 Running Full Pipeline with Cerebrus Tracking System v9")
            run_full_pipeline(args.video_path, args.output_dir, args.player_id, "Cerebrus")
        else:
            print("🎯 Running Full Traditional Pipeline")
            run_full_pipeline(args.video_path, args.output_dir, args.player_id, "Traditional")
    elif args.stage == "detection":
        run_detection_only(args.video_path, args.output_dir)
    elif args.stage == "tracking":
        if not args.detections:
            print("❌ Error: --detections required for tracking stage")
            return
        run_tracking_only(args.detections, args.output_dir)
    elif args.stage == "cerebrus":
        print("🎯 Running Cerebrus Tracking System v9 Only")
        run_cerebrus_only(args.video_path, args.output_dir, args.player_id)

if __name__ == "__main__":
    main()
