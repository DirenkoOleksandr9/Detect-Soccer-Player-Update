#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
This helps catch import errors before running the full pipeline
"""

def test_imports():
    """Test all the imports used in the main pipeline"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import argparse
        import os
        import json
        print("✅ Basic imports successful")
        
        # Test our custom modules
        from player_detection import SoccerPlayerDetector
        print("✅ SoccerPlayerDetector import successful")
        
        from bytetrack_tracker import process_tracking
        print("✅ process_tracking import successful")
        
        from reid_system import AdvancedPlayerReID
        print("✅ AdvancedPlayerReID import successful")
        
        from motion_compensation import apply_gmc_to_pipeline
        print("✅ apply_gmc_to_pipeline import successful")
        
        from stitch_tracklets import stitch_tracklets_offline
        print("✅ stitch_tracklets_offline import successful")
        
        from advanced_event_detection import AdvancedEventDetector
        print("✅ AdvancedEventDetector import successful")
        
        from video_assembly import VideoAssembler
        print("✅ VideoAssembler import successful")
        
        from cerebrus_tracker import CerebrusTracker, process_video_with_cerebrus
        print("✅ CerebrusTracker imports successful")
        
        print("\n🎉 All imports successful! The pipeline should work correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
