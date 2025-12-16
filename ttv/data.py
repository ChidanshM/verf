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
		
		# 1. SAFETY UPDATE: Force load to CPU to save GPU/Pinned RAM
		full_data = torch.load(file_path, map_location='cpu')
		
		# 2. Window Slicing
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
			crop = tensor[..., start : start + window_size]
			if pad_amt > 0:
				crop = torch.nn.functional.pad(crop, (0, pad_amt))
			sliced_data[sensor] = crop

		# 3. Dynamic Mirroring
		if self.mode == "train" and random.random() > 0.5:
			for sensor in sliced_data.keys():
				sliced_data[sensor] = -sliced_data[sensor]

		return sliced_data, label

class BalancedBatchSampler(Sampler):
	def __init__(self, labels, batch_size, samples_per_class=8):
		self.labels = labels
		self.batch_size = batch_size
		self.samples_per_class = samples_per_class
		self.classes_per_batch = self.batch_size // self.samples_per_class
		
		self.label_to_indices = defaultdict(list)
		for idx, label in enumerate(labels):
			self.label_to_indices[label].append(idx)
			
		self.unique_labels = list(self.label_to_indices.keys())
		self.n_batches = int(len(self.unique_labels) // self.classes_per_batch) * 5

	def __iter__(self):
		for _ in range(self.n_batches):
			classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
			indices = []
			for class_ in classes:
				class_indices = self.label_to_indices[class_]
				selected = np.random.choice(class_indices, self.samples_per_class, replace=True)
				indices.extend(selected)
			yield indices

	def __len__(self):
		return self.n_batches

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
	all_files = sorted(list(Path(data_dir).glob("*.pt")))
	if not all_files:
		raise RuntimeError(f"No .pt files found in {data_dir}")

	# Label Extraction
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

	# --- CRITICAL MEMORY FIXES ---
	train_loader = DataLoader(
		train_ds, 
		batch_sampler=sampler, 
		num_workers=0,      # <--- Disables multiprocessing (Low RAM)
		pin_memory=False    # <--- Disables pinned RAM buffer (Low RAM)
	)
	
	val_loader = DataLoader(
		val_ds, 
		batch_size=cfg.batch_size, 
		shuffle=False, 
		num_workers=0,      # <--- Disables multiprocessing
		pin_memory=False    # <--- Disables pinned RAM buffer
	)

	return train_loader, val_loader