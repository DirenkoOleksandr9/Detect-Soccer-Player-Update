#!/usr/bin/env python3
"""
Quick Setup Script for Local Testing
Installs missing packages and tests KPR setup
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 Quick Setup for Local KPR Testing")
    print("=" * 50)
    
    # List of packages to install
    packages = [
        "torch",
        "torchvision", 
        "opencv-python",
        "pandas",
        "filterpy",
        "gdown",
        "yacs",
        "pyyaml"
    ]
    
    print("📦 Installing required packages...")
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    
    print("\n🧪 Testing imports...")
    
    # Test imports
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision not available")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not available")
    
    try:
        import filterpy
        print("✅ FilterPy available")
    except ImportError:
        print("❌ FilterPy not available")
    
    try:
        import gdown
        print("✅ GDown available")
    except ImportError:
        print("❌ GDown not available")
    
    try:
        import yacs
        print("✅ YACS available")
    except ImportError:
        print("❌ YACS not available")
    
    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        print("❌ PyYAML not available")
    
    print("\n🎯 Setup complete! Now you can run:")
    print("   python local_test.py")

if __name__ == "__main__":
    main()
