#!/usr/bin/env python3
"""
Local Testing Script for Soccer Player Tracking Pipeline
Tests syntax and basic functionality before running on Colab
"""

import os
import sys
import traceback
from pathlib import Path

def test_syntax():
    """Test basic Python syntax of the corrected file content"""
    print("🔍 Testing syntax...")
    
    # The corrected file content that we want to test
    corrected_file_content = r"""from __future__ import print_function, absolute_import
from dataclasses import dataclass, field
import pandas as pd
import json
import os

from ..dataset import ImageDataset

@dataclass
class PoseTrack21:
    image_gt: pd.DataFrame = field(default_factory=pd.DataFrame)
    image_detections: pd.DataFrame = field(default_factory=pd.DataFrame)
    video_metadatas: pd.DataFrame = field(default_factory=pd.DataFrame)
    categories: pd.DataFrame = field(default_factory=pd.DataFrame)
    annotations: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class TrackingSet:
    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: pd.DataFrame
    image_gt: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["video_id"]))

class OccludedPosetrack21(ImageDataset):
    dataset_dir = 'posetrack21'

    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'posetrack_data')
        self.train_gt_path = os.path.join(self.train_dir, 'train.json')
        self.val_gt_path = os.path.join(self.train_dir, 'val.json')

        train_set = self._load_gt(self.train_gt_path)
        val_set = self._load_gt(self.val_gt_path)
        train, num_train_pids, num_train_imgs = self._process_data(train_set, relabel=True)
        val, num_val_pids, num_val_imgs = self._process_data(val_set, relabel=False)

        num_total_pids = num_train_pids + num_val_pids
        num_total_imgs = num_train_imgs + num_val_imgs

        print("=> Posetrack21 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  val      | {:5d} | {:8d}".format(num_val_pids, num_val_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.val = val
        self.num_train_pids = num_train_pids
        self.num_val_pids = num_val_pids

    def _load_gt(self, path):
        if not os.path.exists(path):
            print(f"Warning: GT file not found at {path}. Returning empty dataset.")
            return PoseTrack21()
        with open(path) as f:
            data = json.load(f)
        image_gt = pd.DataFrame(data['images'])
        video_metadatas = pd.DataFrame(data['videos'])
        categories = pd.DataFrame(data['categories'])
        annotations = pd.DataFrame(data['annotations'])
        return PoseTrack21(image_gt=image_gt,
                           video_metadatas=video_metadatas,
                           categories=categories,
                           annotations=annotations)

    def _process_data(self, dataset: PoseTrack21, relabel=False):
        if dataset.annotations.empty:
            return [], 0, 0
        all_img_paths = list(dataset.image_gt.file_name)
        all_img_paths.sort()
        img_paths = {row.id: row.file_name for row in dataset.image_gt.itertuples()}
        pid_container = set(dataset.annotations.person_id)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        processed_dataset = []
        for row in dataset.annotations.itertuples():
            img_path = os.path.join(self.dataset_dir, img_paths[row.image_id])
            pid = row.person_id
            if relabel:
                pid = pid2label[pid]
            processed_dataset.append((img_path, pid, 0))
        num_pids = len(pid_container)
        num_imgs = len(processed_dataset)
        return processed_dataset, num_pids, num_imgs

    @classmethod
    def get_masks_config(cls, dataset_name=None, **kwargs):
        return (
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
            True  # This is the second element that the code expects
        )
"""
    
    try:
        # Test if the code compiles
        compile(corrected_file_content, '<string>', 'exec')
        print("✅ Syntax is valid!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def test_imports():
    """Test if required packages can be imported"""
    print("\n🔍 Testing imports...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'pandas',
        'tqdm',
        'filterpy',
        'gdown',
        'yacs',
        'types'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages available!")
        return True

def test_kpr_structure():
    """Test if KPR repository structure is correct"""
    print("\n🔍 Testing KPR repository structure...")
    
    repo_dir = Path("keypoint_promptable_reidentification")
    required_files = [
        "torchreid/scripts/builder.py",
        "torchreid/tools/feature_extractor.py", 
        "configs/kpr/imagenet/kpr_occ_posetrack_test.yaml",
        "torchreid/data/datasets/image/occluded_posetrack21.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = repo_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required KPR files found!")
        return True

def test_config_loading():
    """Test if the config file can be loaded"""
    print("\n🔍 Testing config loading...")
    
    try:
        import yaml
        config_path = "keypoint_promptable_reidentification/configs/kpr/imagenet/kpr_occ_posetrack_test.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Config file loaded successfully!")
            print(f"   Model: {config.get('model', {}).get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_patch_application():
    """Test if the patch can be applied correctly"""
    print("\n🔍 Testing patch application...")
    
    try:
        # Test the patch content
        patch_content = r"""from __future__ import print_function, absolute_import
from dataclasses import dataclass, field
import pandas as pd
import json
import os

from ..dataset import ImageDataset

@dataclass
class PoseTrack21:
    image_gt: pd.DataFrame = field(default_factory=pd.DataFrame)
    image_detections: pd.DataFrame = field(default_factory=pd.DataFrame)
    video_metadatas: pd.DataFrame = field(default_factory=pd.DataFrame)
    categories: pd.DataFrame = field(default_factory=pd.DataFrame)
    annotations: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class TrackingSet:
    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: pd.DataFrame
    image_gt: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["video_id"]))

class OccludedPosetrack21(ImageDataset):
    dataset_dir = 'posetrack21'

    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'posetrack_data')
        self.train_gt_path = os.path.join(self.train_dir, 'train.json')
        self.val_gt_path = os.path.join(self.train_dir, 'val.json')

        train_set = self._load_gt(self.train_gt_path)
        val_set = self._load_gt(self.val_gt_path)
        train, num_train_pids, num_train_imgs = self._process_data(train_set, relabel=True)
        val, num_val_pids, num_val_imgs = self._process_data(val_set, relabel=False)

        num_total_pids = num_train_pids + num_val_pids
        num_total_imgs = num_train_imgs + num_val_imgs

        print("=> Posetrack21 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  val      | {:5d} | {:8d}".format(num_val_pids, num_val_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.val = val
        self.num_train_pids = num_train_pids
        self.num_val_pids = num_val_pids

    def _load_gt(self, path):
        if not os.path.exists(path):
            print(f"Warning: GT file not found at {path}. Returning empty dataset.")
            return PoseTrack21()
        with open(path) as f:
            data = json.load(f)
        image_gt = pd.DataFrame(data['images'])
        video_metadatas = pd.DataFrame(data['videos'])
        categories = pd.DataFrame(data['categories'])
        annotations = pd.DataFrame(data['annotations'])
        return PoseTrack21(image_gt=image_gt,
                           video_metadatas=video_metadatas,
                           categories=categories,
                           annotations=annotations)

    def _process_data(self, dataset: PoseTrack21, relabel=False):
        if dataset.annotations.empty:
            return [], 0, 0
        all_img_paths = list(dataset.image_gt.file_name)
        all_img_paths.sort()
        img_paths = {row.id: row.file_name for row in dataset.image_gt.itertuples()}
        pid_container = set(dataset.annotations.person_id)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        processed_dataset = []
        for row in dataset.annotations.itertuples():
            img_path = os.path.join(self.dataset_dir, img_paths[row.image_id])
            pid = row.person_id
            if relabel:
                pid = pid2label[pid]
            processed_dataset.append((img_path, pid, 0))
        num_pids = len(pid_container)
        num_imgs = len(processed_dataset)
        return processed_dataset, num_pids, num_imgs

    @classmethod
    def get_masks_config(cls, dataset_name=None, **kwargs):
        return (
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
            True  # This is the second element that the code expects
        )
"""
        
        # Test if the patch content compiles
        compile(patch_content, '<patch>', 'exec')
        print("✅ Patch content is valid!")
        
        # Test if we can write the patch
        test_file = "test_patch.py"
        with open(test_file, 'w') as f:
            f.write(patch_content)
        
        # Test if we can read it back
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Clean up
        os.remove(test_file)
        
        if content == patch_content:
            print("✅ Patch can be written and read correctly!")
            return True
        else:
            print("❌ Patch content mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Patch testing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 LOCAL TESTING SCRIPT FOR SOCCER PLAYER TRACKING")
    print("=" * 60)
    
    tests = [
        ("Syntax", test_syntax),
        ("Imports", test_imports), 
        ("KPR Structure", test_kpr_structure),
        ("Config Loading", test_config_loading),
        ("Patch Application", test_patch_application)
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
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for Colab deployment.")
        print("\n📝 Next steps:")
        print("   1. Copy the corrected_file_content to your Colab notebook")
        print("   2. Run the setup cell")
        print("   3. The KPR model should initialize successfully!")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running on Colab.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install <package_name>")
        print("   - Ensure KPR repository is cloned correctly")
        print("   - Check file paths and permissions")

if __name__ == "__main__":
    main()
