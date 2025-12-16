import torch
import numpy as np
import random
import re
import gc
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler 

from .config import Config

class GaitDataset(Dataset):
	def __init__(self, file_paths: list, labels: list, cfg: Config, mode: str = "train"):
		self.file_paths = file_paths
		self.labels = labels
		self.cfg = cfg
		self.mode = mode 

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, idx):
		file_path = self.file_paths[idx]
		label = self.labels[idx]
		
		# 1. Load Data (On-Demand)
		# map_location='cpu' prevents GPU VRAM usage.
		# We wrap in try/except to handle potential corrupted files gracefully
		try:
			full_data = torch.load(file_path, map_location='cpu')
		except Exception as e:
			print(f"Error loading {file_path}: {e}")
			# Return zeros as fallback to prevent crash
			return self._get_empty_window(), label

		# 2. Window Slicing
		# We immediately slice and DELETE 'full_data' to free RAM
		first_sensor = next(iter(full_data.values()))
		seq_len = first_sensor.shape[-1]
		window_size = self.cfg.window_size

		if seq_len < window_size:
			start = 0
			pad_amt = window_size - seq_len
		else:
			if self.mode == "train":
				start = random.randint(0, seq_len - window_size)
			else:
				start = (seq_len - window_size) // 2
			pad_amt = 0

		sliced_data = {}
		for sensor, tensor in full_data.items():
			# Slice and Clone to detach from the big tensor memory
			crop = tensor[..., start : start + window_size].clone()
			
			if pad_amt > 0:
				crop = torch.nn.functional.pad(crop, (0, pad_amt))
			sliced_data[sensor] = crop

		# Explicitly delete big tensor
		del full_data
		
		# 3. Augmentation
		if self.mode == "train" and random.random() > 0.5:
			for sensor in sliced_data.keys():
				sliced_data[sensor] = -sliced_data[sensor]

		return sliced_data, label

	def _get_empty_window(self):
		# Fallback for errors
		dummy = {}
		for s in self.cfg.streams:
			dummy[s] = torch.zeros((1, self.cfg.window_size)) # 1 channel assumption?
		return dummy

class BalancedBatchSampler(Sampler):
	def __init__(self, labels, batch_size, samples_per_class=8):
		self.labels = labels
		self.batch_size = batch_size
		self.samples_per_class = samples_per_class
		
		# Safety: Ensure we don't request more samples than batch allows
		if self.batch_size < self.samples_per_class:
			self.samples_per_class = self.batch_size

		self.classes_per_batch = self.batch_size // self.samples_per_class
		
		self.label_to_indices = defaultdict(list)
		for idx, label in enumerate(labels):
			self.label_to_indices[label].append(idx)
			
		self.unique_labels = list(self.label_to_indices.keys())
		
		# Restore ~15k runs feel: Visit every user ~50 times per epoch
		# Adjusted multiplier to 40 for safety
		self.n_batches = int(len(self.unique_labels) // self.classes_per_batch) * 40

	def __iter__(self):
		for _ in range(self.n_batches):
			# Pick P classes
			classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
			indices = []
			for class_ in classes:
				class_indices = self.label_to_indices[class_]
				# Pick K samples (with replacement if needed)
				selected = np.random.choice(class_indices, self.samples_per_class, replace=True)
				indices.extend(selected)
			yield indices

	def __len__(self):
		return self.n_batches

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
	all_files = sorted(list(Path(data_dir).glob("*.pt")))
	if not all_files:
		raise RuntimeError(f"No .pt files found in {data_dir}")

	# Lightweight Indexing (Just paths and labels, no heavy data)
	labels = []
	valid_files = []
	
	for f in all_files:
		stem = f.name.split('_')[0].split('.')[0]
		numeric_part = re.sub(r'\D', '', stem)
		if numeric_part:
			labels.append(int(numeric_part))
			valid_files.append(f)

	# Split
	unique_ids = sorted(list(set(labels)))
	n_ids = len(unique_ids)
	idx_train = int(n_ids * 0.70)
	idx_val = int(n_ids * 0.85)
	
	train_ids = set(unique_ids[:idx_train])
	val_ids = set(unique_ids[idx_train:idx_val])
	
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
		logger.info(f"Train IDs: {len(train_ids)} | Val IDs: {len(val_ids)}")
		logger.info(f"Train Files: {len(train_paths)} | Val Files: {len(val_paths)}")

	train_ds = GaitDataset(train_paths, train_labels, cfg, mode="train")
	val_ds = GaitDataset(val_paths, val_labels, cfg, mode="val")

	sampler = BalancedBatchSampler(train_labels, batch_size=cfg.batch_size, samples_per_class=8)

	# --- CRITICAL CONFIG FOR STABILITY ---
	# num_workers=0: No multiprocessing overhead. Simplest, safest.
	# pin_memory=False: No page-locking. Saves RAM.
	train_loader = DataLoader(
		train_ds, 
		batch_sampler=sampler, 
		num_workers=0, 
		pin_memory=False 
	)
	
	val_loader = DataLoader(
		val_ds, 
		batch_size=cfg.batch_size, 
		shuffle=False, 
		num_workers=0,
		pin_memory=False
	)

	return train_loader, val_loader