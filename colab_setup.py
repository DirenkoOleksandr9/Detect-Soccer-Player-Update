import os
import subprocess
import sys

def install_requirements():
    print("Installing required packages...")
    
    packages = [
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "easyocr>=1.7.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "scenedetect>=0.6.0",
        "pytorchvideo>=0.1.5",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "albumentations>=1.3.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def setup_colab_environment():
    print("Setting up Colab environment...")
    
    if not os.path.exists('/content'):
        print("This script is designed to run on Google Colab")
        return False
    
    print("✓ Running on Google Colab")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ No GPU detected")
    except ImportError:
        print("✗ PyTorch not available")
    
    return True

def create_project_structure():
    print("Creating project structure...")
    
    directories = [
        "videos",
        "models", 
        "output",
        "datasets"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_yolo_model():
    print("Downloading YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model downloaded")
        return True
    except Exception as e:
        print(f"✗ Failed to download YOLO model: {e}")
        return False

def main():
    print("=== SOCCER HIGHLIGHT PIPELINE - COLAB SETUP ===")
    
    if not setup_colab_environment():
        return
    
    install_requirements()
    create_project_structure()
    download_yolo_model()
    
    print("\n=== SETUP COMPLETED ===")
    print("Next steps:")
    print("1. Upload your video files to the 'videos' directory")
    print("2. Run the main pipeline:")
    print("   python main_pipeline.py videos/your_video.mp4 --player-id 1")
    print("3. Check the 'output' directory for results")

if __name__ == "__main__":
    main()
