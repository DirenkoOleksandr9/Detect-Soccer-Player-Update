# 🎯 COMPLETE FIX FOR SOCCER NOTEBOOK

## 🔍 ISSUES IDENTIFIED IN YOUR ERROR LOG:

1. **❌ KPR model loading fails**: `weights_only` error with pickle
2. **❌ KalmanFilter error**: `'KalmanFilter' object has no attribute 'initiate'`
3. **❌ get_masks_config error**: Returns wrong format

## 🔧 COMPLETE FIXES:

### FIX 1: KPR Model Loading Issue
**Add this line to your KPR_FeatureExtractor.__init__ method:**
```python
import torch.serialization
torch.serialization.add_safe_globals(['scalar'])
```

### FIX 2: KalmanFilter Initiate Issue
**Replace your STrack.__init__ method with:**
```python
def __init__(self, tlwh, score):
    self._tlwh = np.asarray(tlwh, dtype=np.float32)
    self.score = score
    self.track_id = 0
    self.state = 'new'
    self.is_activated = False
    self.frame_id = 0
    self.start_frame = 0
    self.time_since_update = 0
    
    # FIX: Initialize Kalman Filter properly
    self.kalman_filter = self.init_kalman_filter()
    
    # FIX: Initialize mean and covariance manually instead of using initiate
    initial_state = self.tlwh_to_xyah(self._tlwh)
    self.mean = np.array([initial_state[0], initial_state[1], initial_state[2], initial_state[3], 0, 0, 0, 0], dtype=np.float32)
    self.covariance = np.eye(8) * 10
```

### FIX 3: get_masks_config Issue
**Replace your get_masks_config method with:**
```python
@classmethod
def get_masks_config(cls, dataset_name=None, **kwargs):
    # FIXED: Return the values in the format the KPR library expects
    # The library expects: mask_config[0] = parts_num (int), mask_config[1] = prompt_parts_num (int)
    return (1, 1)  # parts_num, prompt_parts_num
```

## 📝 QUICK APPLY METHOD:

**Copy this complete setup code to replace your existing setup:**

```python
# --- Complete Fix Setup ---

# 1. Apply the corrected file content with the FIXED get_masks_config
corrected_file_content = """from __future__ import print_function, absolute_import
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
        # FIXED: Return the values in the format the KPR library expects
        # The library expects: mask_config[0] = parts_num (int), mask_config[1] = prompt_parts_num (int)
        return (1, 1)  # parts_num, prompt_parts_num
"""

# 2. Apply the patch
file_to_patch = os.path.join(REPO_DIR, 'torchreid/data/datasets/image/occluded_posetrack21.py')
try:
    with open(file_to_patch, 'w') as f:
        f.write(corrected_file_content)
    print("✅ Patch with FIXED get_masks_config method applied successfully.")
except Exception as e:
    print(f"❌ Patch failed: {e}")

# 3. Add safe globals for torch.load
import torch.serialization
torch.serialization.add_safe_globals(['scalar'])
print("✅ Added safe globals for torch.load")
```

## 🚀 READY TO DEPLOY!

**Steps:**
1. Copy the complete setup code above
2. Replace your existing setup code in the notebook
3. Run the notebook - all errors should be fixed!

**The fixes address:**
- ✅ KPR model loading with pickle safety
- ✅ KalmanFilter initialization without 'initiate'
- ✅ get_masks_config returning correct format (1, 1)
