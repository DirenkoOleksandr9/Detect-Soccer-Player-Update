#!/usr/bin/env python3
"""
Test script for the new flexible event detection system.
This will show how much more events are detected compared to the rigid system.
"""

import json
import os
from flexible_event_detection import AdaptiveEventDetector, filter_player_events

def test_event_detection():
    """Test the new flexible event detection system."""
    
    # Check if we have the required files
    if not os.path.exists('/content/output/long_player_track.json'):
        print("❌ long_player_track.json not found. Run the pipeline first.")
        return
    
    if not os.path.exists('/content/videos'):
        print("❌ Video directory not found. Run the pipeline first.")
        return
    
    # Find video file
    video_files = [f for f in os.listdir('/content/videos') if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("❌ No video files found in /content/videos")
        return
    
    video_path = f'/content/videos/{video_files[0]}'
    print(f"🎬 Testing with video: {video_files[0]}")
    
    # Get video dimensions
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"📐 Video dimensions: {video_width}x{video_height}")
    print(f"⏱️ FPS: {fps}")
    
    # Test the new flexible event detector
    print("\n🎯 Testing NEW Flexible Event Detection...")
    flexible_detector = AdaptiveEventDetector(video_width, video_height)
    
    # Detect events with flexible thresholds
    events = flexible_detector.detect_events(
        '/content/output/long_player_track.json', 
        video_path, 
        '/content/output/events_flexible.json'
    )
    
    print(f"✅ Flexible detection found {len(events)} events!")
    
    # Filter for target player
    target_player_id = 1
    player_events = filter_player_events(
        '/content/output/events_flexible.json', 
        target_player_id, 
        '/content/output/player_events_flexible.json'
    )
    
    print(f"🎯 Player {target_player_id} events: {len(player_events)}")
    
    # Show event types
    event_types = {}
    for event in events:
        event_type = event.get('event_type', 'unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print(f"\n📊 Event Types Detected:")
    for event_type, count in sorted(event_types.items()):
        print(f"   {event_type}: {count}")
    
    # Compare with old system if it exists
    if os.path.exists('/content/output/events.json'):
        with open('/content/output/events.json', 'r') as f:
            old_events = json.load(f)
        print(f"\n📈 IMPROVEMENT: Old system: {len(old_events)} events, New system: {len(events)} events")
        print(f"🚀 That's a {len(events)/max(len(old_events), 1):.1f}x improvement!")
    
    return events, player_events

if __name__ == "__main__":
    print("🧪 Testing Flexible Event Detection System")
    print("=" * 50)
    
    try:
        events, player_events = test_event_detection()
        print(f"\n🎉 Test completed successfully!")
        print(f"📊 Total events: {len(events)}")
        print(f"🎯 Player events: {len(player_events)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
