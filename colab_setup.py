import os
import subprocess
import sys
import numpy as np
from kalman_filter import KalmanFilter

# FIXED STrack class for Colab notebook
# Copy this into your Colab notebook to replace the broken STrack class

class STrack():
    """A single tracked object with state managed by a Kalman Filter."""
    def __init__(self, tlwh, score):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.kalman_filter = self.init_kalman_filter()
        
        # Initialize Kalman Filter state manually (FIXED)
        initial_state = self.tlwh_to_xyah(self.tlwh)
        self.mean = np.array([initial_state[0], initial_state[1], initial_state[2], initial_state[3], 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8) * 10

        self.track_id = 0
        self.state = 'new'
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0

    def init_kalman_filter(self):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
                        [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]])
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        return kf

    def tlwh_to_xyah(self, tlwh):
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, detection_tlwh, score):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(detection_tlwh))
        self.score = score
        self.state = 'tracked'
        self.is_activated = True
        self.time_since_update = 0

    def activate(self, frame_id, track_id):
        self.track_id = track_id
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = 'tracked'
        self.is_activated = True

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

print("✅ Fixed STrack class ready to copy into Colab notebook")

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
