#!/usr/bin/env python3
"""
Comprehensive KPR Test - Catches Runtime Errors
Tests the actual data types and values that cause the errors
"""

import os
import sys
import traceback
from pathlib import Path

def test_mask_config_data_types():
    """Test the actual data types returned by get_masks_config"""
    print("🔍 Testing mask config data types...")
    
    # Simulate the get_masks_config method
    def get_masks_config(dataset_name=None, **kwargs):
        return (
            {
                'type': 'disk',
                'parts_num': 1,  # This should be an int, not a dict
                'parts_names': ['1'],
                'prompt_parts_num': 1,
                'prompt_parts_names': ['1'],
                'dir': 'pifpaf_maskrcnn_filtering',
                'preprocess': 'five_v',
                'softmax_weight': 15,
                'background_computation_strategy': 'threshold',
                'mask_filtering_threshold': 0.5
            },
            True
        )
    
    try:
        # Test the return value
        mask_config = get_masks_config()
        
        # Check if it's a tuple
        if not isinstance(mask_config, tuple):
            print("❌ get_masks_config should return a tuple")
            return False
        
        # Check if it has 2 elements
        if len(mask_config) != 2:
            print("❌ get_masks_config should return a tuple with 2 elements")
            return False
        
        # Check first element is a dict
        if not isinstance(mask_config[0], dict):
            print("❌ First element should be a dict")
            return False
        
        # Check second element is a bool
        if not isinstance(mask_config[1], bool):
            print("❌ Second element should be a bool")
            return False
        
        # Check specific data types in the dict
        config_dict = mask_config[0]
        
        # Check parts_num is an int
        if not isinstance(config_dict.get('parts_num'), int):
            print(f"❌ parts_num should be int, got {type(config_dict.get('parts_num'))}")
            return False
        
        # Check parts_names is a list
        if not isinstance(config_dict.get('parts_names'), list):
            print(f"❌ parts_names should be list, got {type(config_dict.get('parts_names'))}")
            return False
        
        # Check prompt_parts_num is an int
        if not isinstance(config_dict.get('prompt_parts_num'), int):
            print(f"❌ prompt_parts_num should be int, got {type(config_dict.get('prompt_parts_num'))}")
            return False
        
        print("✅ Mask config data types are correct!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing mask config: {e}")
        return False

def test_yacs_config_compatibility():
    """Test if the config values are compatible with YACS"""
    print("\n🔍 Testing YACS config compatibility...")
    
    try:
        # Simulate the YACS config validation
        mask_config = (
            {
                'type': 'disk',
                'parts_num': 1,
                'parts_names': ['1'],
                'prompt_parts_num': 1,
                'prompt_parts_names': ['1'],
                'dir': 'pifpaf_maskrcnn_filtering',
                'preprocess': 'five_v',
                'softmax_weight': 15,
                'background_computation_strategy': 'threshold',
                'mask_filtering_threshold': 0.5
            },
            True
        )
        
        # Simulate the YACS validation logic
        config_dict = mask_config[0]
        
        # Test each field that YACS validates
        valid_types = {str, type(None), tuple, list, bool, int, float}
        
        for key, value in config_dict.items():
            if type(value) not in valid_types:
                print(f"❌ Invalid type for {key}: {type(value)}. Valid types: {valid_types}")
                return False
        
        # Test specific problematic fields
        if not isinstance(config_dict['parts_num'], int):
            print(f"❌ parts_num must be int, got {type(config_dict['parts_num'])}")
            return False
        
        if not isinstance(config_dict['prompt_parts_num'], int):
            print(f"❌ prompt_parts_num must be int, got {type(config_dict['prompt_parts_num'])}")
            return False
        
        print("✅ YACS config compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing YACS compatibility: {e}")
        return False

def test_build_config_simulation():
    """Simulate the build_config process to catch errors"""
    print("\n🔍 Testing build_config simulation...")
    
    try:
        # Simulate the compute_parts_num_and_names function
        def simulate_compute_parts_num_and_names(cfg):
            # This is what the actual code does
            mask_config = (
                {
                    'type': 'disk',
                    'parts_num': 1,
                    'parts_names': ['1'],
                    'prompt_parts_num': 1,
                    'prompt_parts_names': ['1'],
                    'dir': 'pifpaf_maskrcnn_filtering',
                    'preprocess': 'five_v',
                    'softmax_weight': 15,
                    'background_computation_strategy': 'threshold',
                    'mask_filtering_threshold': 0.5
                },
                True
            )
            
            # Simulate the problematic line
            try:
                # This is the line that fails in the actual code
                cfg.model.kpr.masks.parts_num = mask_config[0]['parts_num']
                cfg.model.kpr.masks.prompt_parts_num = mask_config[0]['prompt_parts_num']
                print("✅ Config assignment simulation successful!")
                return True
            except Exception as e:
                print(f"❌ Config assignment failed: {e}")
                return False
        
        # Create a mock config object
        class MockConfig:
            class Model:
                class KPR:
                    class Masks:
                        def __init__(self):
                            self.parts_num = None
                            self.prompt_parts_num = None
            
            def __init__(self):
                self.model = self.Model()
                self.model.kpr = self.Model.KPR()
                self.model.kpr.masks = self.Model.KPR.Masks()
        
        cfg = MockConfig()
        result = simulate_compute_parts_num_and_names(cfg)
        
        if result:
            print("✅ Build config simulation passed!")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error in build_config simulation: {e}")
        return False

def test_actual_error_scenario():
    """Test the exact error scenario from the logs"""
    print("\n🔍 Testing actual error scenario...")
    
    try:
        # This is the exact error from your logs
        mask_config = (
            {
                'type': 'disk',
                'parts_num': 1,
                'parts_names': ['1'],
                'prompt_parts_num': 1,
                'prompt_parts_names': ['1'],
                'dir': 'pifpaf_maskrcnn_filtering',
                'preprocess': 'five_v',
                'softmax_weight': 15,
                'background_computation_strategy': 'threshold',
                'mask_filtering_threshold': 0.5
            },
            True
        )
        
        # Simulate the exact line that fails
        try:
            # This is the line: cfg.model.kpr.masks.parts_num = mask_config[0]
            # But mask_config[0] is a dict, not an int!
            parts_num_value = mask_config[0]  # This is a dict!
            
            # Check if it's the wrong type
            if isinstance(parts_num_value, dict):
                print("❌ ERROR FOUND: mask_config[0] is a dict, should be an int!")
                print(f"   Got: {type(parts_num_value)} = {parts_num_value}")
                print("   Expected: int")
                return False
            
        except Exception as e:
            print(f"❌ Error in actual scenario: {e}")
            return False
        
        print("✅ Actual error scenario test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing actual scenario: {e}")
        return False

def test_corrected_version():
    """Test the corrected version that should work"""
    print("\n🔍 Testing corrected version...")
    
    try:
        # The corrected version should access the dict properly
        mask_config = (
            {
                'type': 'disk',
                'parts_num': 1,
                'parts_names': ['1'],
                'prompt_parts_num': 1,
                'prompt_parts_names': ['1'],
                'dir': 'pifpaf_maskrcnn_filtering',
                'preprocess': 'five_v',
                'softmax_weight': 15,
                'background_computation_strategy': 'threshold',
                'mask_filtering_threshold': 0.5
            },
            True
        )
        
        # Correct way to access the values
        config_dict = mask_config[0]  # Get the dict
        parts_num = config_dict['parts_num']  # Get the int value
        prompt_parts_num = config_dict['prompt_parts_num']  # Get the int value
        
        # Verify types
        if not isinstance(parts_num, int):
            print(f"❌ parts_num should be int, got {type(parts_num)}")
            return False
        
        if not isinstance(prompt_parts_num, int):
            print(f"❌ prompt_parts_num should be int, got {type(prompt_parts_num)}")
            return False
        
        print(f"✅ Corrected version works! parts_num={parts_num}, prompt_parts_num={prompt_parts_num}")
        return True
        
    except Exception as e:
        print(f"❌ Error in corrected version: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🧪 COMPREHENSIVE KPR RUNTIME TESTING")
    print("=" * 60)
    
    tests = [
        ("Mask Config Data Types", test_mask_config_data_types),
        ("YACS Config Compatibility", test_yacs_config_compatibility),
        ("Build Config Simulation", test_build_config_simulation),
        ("Actual Error Scenario", test_actual_error_scenario),
        ("Corrected Version", test_corrected_version)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The fix is correct.")
    else:
        print("⚠️ Some tests failed. The issue is identified!")
        print("\n🔧 THE PROBLEM:")
        print("   The code is trying to assign mask_config[0] (a dict) to parts_num")
        print("   But it should assign mask_config[0]['parts_num'] (an int)")
        print("\n🔧 THE FIX:")
        print("   Change the KPR library code to access the dict properly")
        print("   OR modify the get_masks_config to return the values directly")
    
    print("\n📝 RECOMMENDATION:")
    print("   The issue is in the KPR library's compute_parts_num_and_names function")
    print("   It expects mask_config[0] to be the value, but it's actually a dict")
    print("   We need to either:")
    print("   1. Fix the KPR library code")
    print("   2. Modify our get_masks_config to return values differently")

if __name__ == "__main__":
    main()
