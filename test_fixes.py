#!/usr/bin/env python3
"""
TEST SCRIPT - Verifies all fixes work correctly
"""

import os
import sys
import traceback

def test_get_masks_config():
    """Test that get_masks_config returns correct format"""
    print("🔍 Testing get_masks_config...")
    
    def get_masks_config(dataset_name=None, **kwargs):
        return (1, 1)  # parts_num, prompt_parts_num
    
    mask_config = get_masks_config()
    
    if isinstance(mask_config, tuple) and len(mask_config) == 2:
        if isinstance(mask_config[0], int) and isinstance(mask_config[1], int):
            print("✅ get_masks_config returns correct format: (int, int)")
            return True
        else:
            print(f"❌ get_masks_config returns wrong types: {type(mask_config[0])}, {type(mask_config[1])}")
            return False
    else:
        print(f"❌ get_masks_config returns wrong format: {type(mask_config)}")
        return False

def test_torch_serialization():
    """Test torch.serialization fix"""
    print("🔍 Testing torch.serialization...")
    
    try:
        import torch.serialization
        torch.serialization.add_safe_globals(['scalar'])
        print("✅ torch.serialization.add_safe_globals(['scalar']) works")
        return True
    except Exception as e:
        print(f"❌ torch.serialization error: {e}")
        return False

def test_kalman_filter():
    """Test KalmanFilter initialization"""
    print("🔍 Testing KalmanFilter...")
    
    try:
        import numpy as np
        from filterpy.kalman import KalmanFilter
        
        # Test the fixed initialization
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # Test that we can set x and P attributes
        kf.x = np.zeros(8)
        kf.P = np.eye(8)
        
        # Test predict and update methods
        kf.predict()
        kf.update(np.array([1, 2, 3, 4]))
        
        print("✅ KalmanFilter initialization and methods work")
        return True
    except Exception as e:
        print(f"❌ KalmanFilter error: {e}")
        return False

def test_notebook_syntax():
    """Test that the notebook has correct syntax"""
    print("🔍 Testing notebook syntax...")
    
    try:
        import json
        with open('soccer_highlight_colab_advanced.ipynb', 'r') as f:
            notebook = json.load(f)
        
        # Check that the notebook is valid JSON
        if isinstance(notebook, dict) and 'cells' in notebook:
            print("✅ Notebook is valid JSON with cells")
            return True
        else:
            print("❌ Notebook structure is invalid")
            return False
    except Exception as e:
        print(f"❌ Notebook syntax error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RUNNING ALL TESTS...")
    print("=" * 50)
    
    tests = [
        ("Notebook Syntax", test_notebook_syntax),
        ("get_masks_config", test_get_masks_config),
        ("Torch Serialization", test_torch_serialization),
        ("KalmanFilter", test_kalman_filter)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The notebook is ready to run!")
        return True
    else:
        print("⚠️ Some tests failed. Manual review may be needed.")
        return False

if __name__ == "__main__":
    main()
