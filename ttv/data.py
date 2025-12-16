import torch
import numpy as np
import random
import re
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler 

from .config import Config

class GaitDataset(Dataset):
	def __init__(self, data_dir: str, cfg: Config, file_paths: list, labels: list, mode: str = "train"):
		self.cfg = cfg
		self.file_paths = file_paths
		self.labels = labels
		self.data_dir = Path(data_dir)
		self.mode = mode 

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, idx):
		file_path = self.file_paths[idx]
		label = self.labels[idx]
		
		# 1. Load FULL Session
		full_data = torch.load(file_path)
		
		# 2. RESTORED: Window Slicing (Fixes the crash)
		# We need to cut exactly 'window_size' (e.g., 200) from the long signal.
		first_sensor = next(iter(full_data.values()))
		seq_len = first_sensor.shape[-1]
		window_size = self.cfg.window_size # e.g. 200

		# Safety check for short files
		if seq_len < window_size:
			start = 0
			pad_amt = window_size - seq_len
		else:
			# Random crop for training, Center for val
			if self.mode == "train":
				start = random.randint(0, seq_len - window_size)
			else:
				start = (seq_len - window_size) // 2
			pad_amt = 0

		# Slice all sensors consistently
		sliced_data = {}
		for sensor, tensor in full_data.items():
			# [Channels, Time] -> Slice Time dimension
			crop = tensor[..., start : start + window_size]
			
			if pad_amt > 0:
				crop = torch.nn.functional.pad(crop, (0, pad_amt))
				
			sliced_data[sensor] = crop

		# 3. Dynamic Mirroring (Train only)
		if self.mode == "train" and random.random() > 0.5:
			for sensor in sliced_data.keys():
				sliced_data[sensor] = -sliced_data[sensor]

		return sliced_data, label

class BalancedBatchSampler(Sampler):
	"""
	Ensures P identities and K samples per batch.
	"""
	def __init__(self, labels, batch_size, samples_per_class=8):
		self.labels = labels
		self.batch_size = batch_size
		self.samples_per_class = samples_per_class
		self.classes_per_batch = self.batch_size // self.samples_per_class
		
		self.label_to_indices = defaultdict(list)
		for idx, label in enumerate(labels):
			self.label_to_indices[label].append(idx)
			
		self.unique_labels = list(self.label_to_indices.keys())
		
		# Define epoch length: Visit every user ~5 times per epoch (since we reuse files)
		self.n_batches = int(len(self.unique_labels) // self.classes_per_batch) * 5

	def __iter__(self):
		for _ in range(self.n_batches):
			classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
			indices = []
			for class_ in classes:
				class_indices = self.label_to_indices[class_]
				# replace=True allows us to get 8 samples from 1 file (by cropping differently each time)
				selected = np.random.choice(class_indices, self.samples_per_class, replace=True)
				indices.extend(selected)
			yield indices

	def __len__(self):
		return self.n_batches

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
	all_files = sorted(list(Path(data_dir).glob("*.pt")))
	if not all_files:
		raise RuntimeError(f"No .pt files found in {data_dir}")

	# Robust Label Extraction (Handles S001.pt)
	labels = []
	valid_files = []
	for f in all_files:
		stem = f.name.split('_')[0].split('.')[0]
		numeric_part = re.sub(r'\D', '', stem)
		if numeric_part:
			labels.append(int(numeric_part))
			valid_files.append(f)

	# --- 3-WAY SPLIT (70% Train, 15% Val, 15% Test) ---
	unique_ids = sorted(list(set(labels)))
	n_ids = len(unique_ids)
	
	idx_train = int(n_ids * 0.70)
	idx_val = int(n_ids * 0.85)
	
	train_ids = set(unique_ids[:idx_train])
	val_ids = set(unique_ids[idx_train:idx_val])
	test_ids = set(unique_ids[idx_val:])
	
	train_paths, train_labels = [], []
	val_paths, val_labels = [], []
	
	for f, l in zip(valid_files, labels):
		if l in train_ids:
			train_paths.append(f)
			train_labels.append(l)
		elif l in val_ids:
			val_paths.append(f)
			val_labels.append(l)

	if logger:
		logger.info(f"Split :: Train: {len(train_ids)} IDs | Val: {len(val_ids)} IDs | Test: {len(test_ids)} IDs")
		logger.info(f"Files :: Train: {len(train_paths)} | Val: {len(val_paths)}")

	train_ds = GaitDataset(data_dir, cfg, train_paths, train_labels, mode="train")
	val_ds = GaitDataset(data_dir, cfg, val_paths, val_labels, mode="val")

	sampler = BalancedBatchSampler(train_labels, batch_size=cfg.batch_size, samples_per_class=8)

	train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

	return train_loader, val_loader