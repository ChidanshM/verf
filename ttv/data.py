import torch
import numpy as np
import random
import re
import gc
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler 
from tqdm import tqdm

from .config import Config

class GaitDataset(Dataset):
	def __init__(self, samples: list, cfg: Config, mode: str = "train"):
		# samples is a list of (tensor_data, label) tuples
		self.samples = samples 
		self.cfg = cfg
		self.mode = mode 

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		# 1. Get pre-loaded window from RAM
		data, label = self.samples[idx]
		
		# 2. Clone to avoid modifying the cached version
		# (Necessary because we flip signs for augmentation)
		window_data = {k: v.clone() for k, v in data.items()}

		# 3. Dynamic Mirroring (Augmentation)
		if self.mode == "train" and random.random() > 0.5:
			for sensor in window_data.keys():
				window_data[sensor] = -window_data[sensor]

		return window_data, label

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
		# Epoch length: Visit every user ~5 times
		self.n_batches = int(len(self.unique_labels) // self.classes_per_batch) * 5

	def __iter__(self):
		for _ in range(self.n_batches):
			classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
			indices = []
			for class_ in classes:
				class_indices = self.label_to_indices[class_]
				# We have many windows per class now, so replacement is less critical but kept for safety
				selected = np.random.choice(class_indices, self.samples_per_class, replace=True)
				indices.extend(selected)
			yield indices

	def __len__(self):
		return self.n_batches

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
	all_files = sorted(list(Path(data_dir).glob("*.pt")))
	if not all_files:
		raise RuntimeError(f"No .pt files found in {data_dir}")

	if logger: logger.info("Pre-slicing data into RAM (Low Memory Mode)...")
	
	# Store all tiny windows here: [(window_dict, label), ...]
	all_samples = []
	
	# Stride: How much to step? 
	# Smaller stride = More data, overlapping windows. 
	# Use 50% overlap for good data density.
	slice_stride = cfg.window_size // 2 
	
	valid_labels = []

	for f in tqdm(all_files, desc="Chunking Files"):
		# 1. Parse Label
		stem = f.name.split('_')[0].split('.')[0]
		numeric_part = re.sub(r'\D', '', stem)
		if not numeric_part: continue
		label = int(numeric_part)
		
		# 2. Load File (CPU)
		try:
			full_data = torch.load(f, map_location='cpu')
		except:
			continue
			
		# 3. Slice into tiny chunks
		first_sensor = next(iter(full_data.values()))
		seq_len = first_sensor.shape[-1]
		
		# Generate windows
		for start in range(0, seq_len - cfg.window_size + 1, slice_stride):
			window = {}
			for k, v in full_data.items():
				# Slice and CLONE to detach from the big file memory
				window[k] = v[..., start : start + cfg.window_size].clone()
			
			all_samples.append((window, label))
			valid_labels.append(label)
			
		# 4. Aggressive Cleanup
		del full_data
		# Force Python to free memory NOW, don't wait
		# This prevents the "Accumulation Crash"
		if len(all_samples) % 100 == 0:
			gc.collect()

	# --- Split Logic ---
	unique_ids = sorted(list(set(valid_labels)))
	n_ids = len(unique_ids)
	idx_train = int(n_ids * 0.70)
	idx_val = int(n_ids * 0.85)
	
	train_ids = set(unique_ids[:idx_train])
	val_ids = set(unique_ids[idx_train:idx_val])
	
	train_data = []
	val_data = []
	
	# Separate labels for sampler
	train_labels_list = []
	
	for sample in all_samples:
		_, lbl = sample
		if lbl in train_ids:
			train_data.append(sample)
			train_labels_list.append(lbl)
		elif lbl in val_ids:
			val_data.append(sample)

	if logger:
		logger.info(f"Generated {len(all_samples)} windows in RAM.")
		logger.info(f"Train Windows: {len(train_data)} | Val Windows: {len(val_data)}")

	train_ds = GaitDataset(train_data, cfg, mode="train")
	val_ds = GaitDataset(val_data, cfg, mode="val")

	sampler = BalancedBatchSampler(train_labels_list, batch_size=cfg.batch_size, samples_per_class=8)

	# Fast loading (Memory is already ready)
	train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

	return train_loader, val_loader