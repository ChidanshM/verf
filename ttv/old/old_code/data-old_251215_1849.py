# verf/ttv/data.py
from __future__ import annotations

import os
import glob
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import Config

class GaitDataset(Dataset):
    """
    Returns (InputDict, Label) instead of Triplet.
    """
    def __init__(self, data_dir: str, cfg: Config, file_paths: list, labels: list):
        self.cfg = cfg
        self.file_paths = file_paths
        self.labels = labels
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load
        data = torch.load(file_path)
        
        # Dynamic Mirroring (Data Augmentation)
        if random.random() > 0.5:
            for sensor in data.keys():
                data[sensor] = -data[sensor]

        return data, label

class BalancedBatchSampler(Sampler):
    """
    Ensures every batch contains P identities and K samples per identity.
    Batch Size = P * K
    """
    def __init__(self, labels, batch_size, samples_per_class=8):
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # P = Batch / K
        self.classes_per_batch = self.batch_size // self.samples_per_class
        
        # Index data by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
            
        self.unique_labels = list(self.label_to_indices.keys())
        
        # Calculate length
        self.n_samples = len(labels)
        self.n_batches = self.n_samples // self.batch_size

    def __iter__(self):
        count = 0
        while count + self.batch_size <= self.n_samples:
            # 1. Pick P random classes
            classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
            
            indices = []
            for class_ in classes:
                # 2. Pick K samples for this class (allow replacement if class is small)
                class_indices = self.label_to_indices[class_]
                replace_flag = len(class_indices) < self.samples_per_class
                selected = np.random.choice(class_indices, self.samples_per_class, replace=replace_flag)
                indices.extend(selected)
                
            yield indices
            count += self.batch_size

    def __len__(self):
        return self.n_batches

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
    # Gather files
    all_files = sorted(list(Path(data_dir).glob("*.pt")))
    if not all_files:
        raise RuntimeError(f"No .pt files found in {data_dir}")

    # Extract Labels (Assuming format: "id_session_seq.pt")
    labels = [int(f.name.split('_')[0]) for f in all_files]
    
    # Train/Val Split (80/20 by Subject ID)
    unique_ids = sorted(list(set(labels)))
    split_idx = int(len(unique_ids) * 0.8)
    
    train_ids = set(unique_ids[:split_idx])
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    for f, l in zip(all_files, labels):
        if l in train_ids:
            train_paths.append(f)
            train_labels.append(l)
        else:
            val_paths.append(f)
            val_labels.append(l)

    if logger:
        logger.info(f"Train IDs: {len(train_ids)} | Val IDs: {len(unique_ids) - len(train_ids)}")
        logger.info(f"Train Files: {len(train_paths)} | Val Files: {len(val_paths)}")

    # Create Datasets
    train_ds = GaitDataset(data_dir, cfg, train_paths, train_labels)
    val_ds = GaitDataset(data_dir, cfg, val_paths, val_labels)

    # Sampler: K=8 samples per person.
    # Note: If batch_size=64, then P=8 people.
    sampler = BalancedBatchSampler(train_labels, batch_size=cfg.batch_size, samples_per_class=8)

    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)
    
    # Validation uses standard shuffling
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# class SixStreamGaitDataset(Dataset):
# 	"""
# 	Returns triplets (anchor, positive, negative) where each element is a dict:
# 		{stream_name: Tensor[C, T]}
# 	For mode="test": returns (anchor_dict, subject_id)
# 	"""

# 	def __init__(
# 		self,
# 		subjects_data: Dict[str, Dict[str, torch.Tensor]],
# 		cfg: Config,
# 		mode: str = "train",
# 		override_stride: Optional[int] = None,
# 	):
# 		self.cfg = cfg
# 		self.mode = mode
# 		self.window_size = cfg.window_size
# 		self.stride = override_stride if override_stride is not None else cfg.stride  #  keep override
# 		self.sensors = list(cfg.streams)

# 		self.data = subjects_data
# 		self.subject_ids = list(subjects_data.keys())
# 		self.samples = []

# 		for subj_id in self.subject_ids:
# 			# use the first sensor as reference length
# 			n_points = len(subjects_data[subj_id][self.sensors[0]])
# 			for start in range(0, n_points - self.window_size, self.stride):
# 				self.samples.append((subj_id, start))

# 	def __len__(self) -> int:
# 		return len(self.samples)
	
# 	def _time_warp(self, tensor_2d: torch.Tensor) -> torch.Tensor:
# 		"""
# 		Randomly stretches or compresses the signal along the time axis.
# 		Input: [Channels, Time] -> Output: [Channels, WindowSize]
# 		"""
# 		# 1. Create a random scale factor (e.g., 0.8x to 1.2x speed)
# 		scale = 0.8 + (torch.rand(1).item() * 0.4)
		
# 		# 2. Resize needs [Batch, Channels, Time], so we fake a batch dim
# 		x = tensor_2d.unsqueeze(0) 
# 		new_len = int(self.window_size * scale)
		
# 		# 3. Interpolate (Stretch/Squash)
# 		x_warped = F.interpolate(x, size=new_len, mode='linear', align_corners=False)
# 		x_warped = x_warped.squeeze(0) # Remove fake batch dim

# 		# 4. Crop or Pad back to original window_size
# 		if new_len > self.window_size:
# 			# If we stretched it (made it longer), crop the center
# 			start = (new_len - self.window_size) // 2
# 			return x_warped[:, start : start + self.window_size]
# 		else:
# 			# If we squashed it (made it shorter), pad with zeros at the end
# 			padding = self.window_size - new_len
# 			return F.pad(x_warped, (0, padding))
		
# 	def _get_window(self, subj_id: str, start_idx: int) -> Dict[str, torch.Tensor]:
# 		window_dict: Dict[str, torch.Tensor] = {}
# 		for sensor in self.sensors:
# 			full_signal = self.data[subj_id][sensor]
# 			signal_slice = full_signal[start_idx : start_idx + self.window_size]

# 			# expected: [T, C] -> [C, T] needed for time warping
# 			# Note: We move the permute UP so we can warp dimensions easier
# 			sensor_data = signal_slice.clone().detach().float().permute(1, 0)

# 			if self.mode == "train":
# 				# 1. Add Noise (Your existing code, adapted for [C, T] shape)
# 				noise = torch.randn_like(sensor_data) * 0.05
# 				sensor_data = sensor_data + noise
				
# 				# 2. Add Time Warp (50% chance per sensor)
# 				if torch.rand(1).item() < 0.8:
# 					sensor_data = self._time_warp(sensor_data)

# 			window_dict[sensor] = sensor_data
			
# 		return window_dict


# 	# def _get_window(self, subj_id: str, start_idx: int) -> Dict[str, torch.Tensor]:
# 	# 	window_dict: Dict[str, torch.Tensor] = {}
# 	# 	for sensor in self.sensors:
# 	# 		full_signal = self.data[subj_id][sensor]
# 	# 		signal_slice = full_signal[start_idx : start_idx + self.window_size]
			
# 	# 		# --- NEW CODE START ---
# 	# 		# Only add noise during training mode
# 	# 		if self.mode == "train":
# 	# 			# Add Gaussian noise (scaled by 0.01 to 0.05 usually works well)
# 	# 			noise = torch.randn_like(signal_slice) * 0.02
# 	# 			signal_slice = signal_slice + noise
				
# 	# 			# Optional: Random Scaling (Simulate sensor intensity drift)
# 	# 			# scale = 0.9 + (torch.rand(1).item() * 0.2) # Random between 0.9 and 1.1
# 	# 			# signal_slice = signal_slice * scale
# 	# 		# --- NEW CODE END ---

# 	# 		# expected: [T, C] -> [C, T]
# 	# 		window_dict[sensor] = signal_slice.clone().detach().float().permute(1, 0)
# 	# 	return window_dict

# 	# def _get_window(self, subj_id: str, start_idx: int) -> Dict[str, torch.Tensor]:
# 	# 	window_dict: Dict[str, torch.Tensor] = {}
# 	# 	for sensor in self.sensors:
# 	# 		full_signal = self.data[subj_id][sensor]
# 	# 		signal_slice = full_signal[start_idx : start_idx + self.window_size]
# 	# 		# expected: [T, C] -> [C, T]
# 	# 		window_dict[sensor] = signal_slice.clone().detach().float().permute(1, 0)
# 	# 	return window_dict

# 	def __getitem__(self, index: int):
# 		anchor_subj, anchor_start = self.samples[index]
# 		anchor_dict = self._get_window(anchor_subj, anchor_start)

# 		if self.mode == "test":
# 			return anchor_dict, anchor_subj

# 		# ... (Positive/Negative mining logic) ...
# 			# Positive: same subject, different window
# 		subj_len = len(self.data[anchor_subj][self.sensors[0]])
# 		pos_start = (
# 			np.random.randint(0, max(1, subj_len - self.window_size))
# 			if subj_len > self.window_size
# 			else anchor_start
# 		)

# 		# Negative: different subject (if possible)
# 		other_subjs = [s for s in self.subject_ids if s != anchor_subj]
# 		neg_subj = np.random.choice(other_subjs) if other_subjs else anchor_subj

# 		neg_len = len(self.data[neg_subj][self.sensors[0]])
# 		neg_start = (
# 			np.random.randint(0, max(1, neg_len - self.window_size))
# 			if neg_len > self.window_size
# 			else 0
# 		)

# 		pos_dict = self._get_window(anchor_subj, pos_start)
# 		neg_dict = self._get_window(neg_subj, neg_start)

# 		#  NEW: Randomly swap Left/Right sensors (Mirroring)
# 		if self.mode == "train" and torch.rand(1).item() < 0.5:
# 			# Helper function to swap dictionary keys
# 			def swap_sensors(d):
# 				# We use temporary variables to swap values safely
# 				d['Shank_LT'], d['Shank_RT'] = d['Shank_RT'], d['Shank_LT']
# 				d['Foot_LT'],  d['Foot_RT']  = d['Foot_RT'],  d['Foot_LT']
# 				return d

# 			# Apply to Anchor and Positive (must match each other!)
# 			anchor_dict = swap_sensors(anchor_dict)
# 			pos_dict    = swap_sensors(pos_dict)
			
# 			# Optional: We usually DON'T swap Negative to keep it a "hard" example,
# 			# but swapping it is also valid.
		
# 		return anchor_dict, pos_dict, neg_dict

# 	# def __getitem__(self, index: int):
# 	# 	anchor_subj, anchor_start = self.samples[index]
# 	# 	anchor_dict = self._get_window(anchor_subj, anchor_start)

# 	# 	if self.mode == "test":
# 	# 		return anchor_dict, anchor_subj

# 	# 	# Positive: same subject, different window
# 	# 	subj_len = len(self.data[anchor_subj][self.sensors[0]])
# 	# 	pos_start = (
# 	# 		np.random.randint(0, max(1, subj_len - self.window_size))
# 	# 		if subj_len > self.window_size
# 	# 		else anchor_start
# 	# 	)
# 	# 	pos_dict = self._get_window(anchor_subj, pos_start)

# 	# 	# Negative: different subject (if possible)
# 	# 	other_subjs = [s for s in self.subject_ids if s != anchor_subj]
# 	# 	neg_subj = np.random.choice(other_subjs) if other_subjs else anchor_subj

# 	# 	neg_len = len(self.data[neg_subj][self.sensors[0]])
# 	# 	neg_start = (
# 	# 		np.random.randint(0, max(1, neg_len - self.window_size))
# 	# 		if neg_len > self.window_size
# 	# 		else 0
# 	# 	)
# 	# 	neg_dict = self._get_window(neg_subj, neg_start)

# 	# 	return anchor_dict, pos_dict, neg_dict


# def create_dataloaders(
# 	data_dir: str,
# 	cfg: Config,
# 	parent_dir: str,
# 	timestamp: str,
# 	logger,
# ) -> Tuple[DataLoader, DataLoader]:
# 	"""
# 	Loads *.pt subject files, splits into train/val/test, writes test IDs to:
# 		parent_dir/test_subjects-{timestamp}.txt
# 	Returns: (train_loader, val_loader)
# 	"""

# 	logger.info(f"Loading .pt files from: {data_dir}")
# 	files = glob.glob(os.path.join(data_dir, "*.pt"))

# 	if not files:
# 		logger.error(f"No .pt files found in {data_dir}!")
# 		raise FileNotFoundError(f"Check your path. Current target: {os.path.abspath(data_dir)}")

# 	np.random.seed(cfg.seed)
# 	np.random.shuffle(files)

# 	master_data = {}
# 	for f in tqdm(files, desc="Loading Data"):
# 		sub_id = os.path.basename(f).split(".")[0]
# 		master_data[sub_id] = torch.load(f)

# 	subject_ids = list(master_data.keys())
# 	n_total = len(subject_ids)
# 	n_train = int(n_total * 0.70)
# 	n_val = int(n_total * 0.15)

# 	train_ids = subject_ids[:n_train]
# 	val_ids = subject_ids[n_train : n_train + n_val]
# 	test_ids = subject_ids[n_train + n_val :]

# 	# Save test IDs so evaluation uses the exact same people
# 	test_ids_path = os.path.join(parent_dir, f"test_subjects-{timestamp}.txt")
# 	with open(test_ids_path, "w") as f:
# 		for sid in test_ids:
# 			f.write(sid + "\n")

# 	logger.info(f"Saved {len(test_ids)} test subjects to {test_ids_path}")
# 	logger.info(f"Split: Train={len(train_ids)}, Val={len(val_ids)}")

# 	def subset_data(ids):
# 		return {k: master_data[k] for k in ids}

# 	train_ds = SixStreamGaitDataset(subset_data(train_ids), cfg=cfg, mode="train")
# 	# For val, your intent seems to be non-overlapping windows => stride = window_size
# 	val_ds = SixStreamGaitDataset(subset_data(val_ids), cfg=cfg, mode="val", override_stride=cfg.window_size)

# 	train_loader = DataLoader(
# 		train_ds,
# 		batch_size=cfg.batch_size,
# 		shuffle=True,
# 		num_workers=4,
# 		pin_memory=True,
# 	)
# 	val_loader = DataLoader(
# 		val_ds,
# 		batch_size=cfg.batch_size,
# 		shuffle=False,
# 		num_workers=4,
# 	)

# 	return train_loader, val_loader
