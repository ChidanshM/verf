import torch
import numpy as np
import random
import re
import shutil
import os
import json
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler 
from tqdm import tqdm

from .config import Config

class WindowDataset(Dataset):
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
		try:
			data = torch.load(file_path, map_location='cpu')
		except Exception:
			return self._get_dummy(), label

		# Z-Score Norm
		for s in data.keys():
			t = data[s]
			m = t.mean(dim=1, keepdim=True)
			std = t.std(dim=1, keepdim=True) + 1e-6
			data[s] = (t - m) / std

		# --- GENTLE AUGMENTATION ---
		if self.mode == "train":
			# 1. Mirroring
			if random.random() > 0.5:
				for s in data.keys():
					data[s] = -data[s]

			# 2. Gentle Scaling (+/- 5%)
			scale = random.uniform(0.95, 1.05)
			for s in data.keys():
				data[s] = data[s] * scale

			# 3. Micro Noise (0.5%)
			for s in data.keys():
				noise = torch.randn_like(data[s]) * 0.005
				data[s] = data[s] + noise

		return data, label

	def _get_dummy(self):
		d = {}
		for s in self.cfg.streams:
			d[s] = torch.zeros((self.cfg.input_channels, self.cfg.window_size))
		return d

class BalancedBatchSampler(Sampler):
	def __init__(self, labels, batch_size, total_windows=None, samples_per_class=8):
		self.labels = labels
		self.batch_size = batch_size
		self.samples_per_class = samples_per_class
		if self.batch_size < self.samples_per_class:
			self.samples_per_class = self.batch_size
		self.classes_per_batch = self.batch_size // self.samples_per_class
		self.label_to_indices = defaultdict(list)
		for idx, label in enumerate(labels):
			self.label_to_indices[label].append(idx)
		self.unique_labels = list(self.label_to_indices.keys())
		if total_windows is None: total_windows = len(labels)
		self.n_batches = max(1, int(total_windows // self.batch_size))

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

def generate_cache(data_dir: Path, cache_dir: Path, cfg: Config, logger):
	if logger: logger.info(f"Generating cache at: {cache_dir}")
	cache_dir.mkdir(parents=True, exist_ok=True)
	all_files = sorted(list(data_dir.glob("*.pt")))
	if not all_files: raise RuntimeError(f"No .pt files found in {data_dir}")

	stride = cfg.window_size // 2
	count = 0
	
	for f in tqdm(all_files, desc=f"Caching Windows"):
		stem = f.name.split('_')[0].split('.')[0]
		try: full_data = torch.load(f, map_location='cpu')
		except: continue
		
		for k, v in full_data.items():
			if v.shape[0] > v.shape[1]: full_data[k] = v.t()
		
		first_val = next(iter(full_data.values()))
		seq_len = first_val.shape[-1]
		
		if seq_len < cfg.window_size: continue

		for i, start in enumerate(range(0, seq_len - cfg.window_size + 1, stride)):
			window = {}
			for k, v in full_data.items():
				window[k] = v[..., start : start + cfg.window_size].clone()
			torch.save(window, cache_dir / f"{stem}_{i:05d}.pt")
			count += 1
		del full_data

	if count == 0: raise RuntimeError(f"Cache generation FAILED! 0 windows created.")
	if logger: logger.info(f"Cache complete. Created {count} windows.")

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
	source_dir = Path(data_dir)
	cache_name = f"cache_win{cfg.window_size}_stride{cfg.stride}"
	cache_dir = source_dir.parent / cache_name 
	
	if not cache_dir.exists() or not any(cache_dir.iterdir()):
		generate_cache(source_dir, cache_dir, cfg, logger)
	else:
		if logger: logger.info(f"Using existing cache: {cache_dir}")

	all_files = sorted(list(cache_dir.glob("*.pt")))
	labels = []
	valid_files = []
	
	for f in all_files:
		stem = f.name.split('_')[0]
		numeric_part = re.sub(r'\D', '', stem)
		if numeric_part:
			labels.append(int(numeric_part))
			valid_files.append(f)

	# --- SHUFFLING & SPLIT LOGIC ---
	unique_ids = sorted(list(set(labels))) # Start sorted
	
	# SHUFFLE Deterministically using seed
	# This ensures a random distribution of subjects (preventing order bias)
	# but guarantees the SAME split if you run the code twice.
	rng = random.Random(cfg.seed)
	rng.shuffle(unique_ids)

	n_ids = len(unique_ids)
	idx_train = int(n_ids * 0.70)
	idx_val = int(n_ids * 0.85)
	
	train_ids = set(unique_ids[:idx_train])
	val_ids = set(unique_ids[idx_train:idx_val])
	test_ids = set(unique_ids[idx_val:])
	
	# Save Split Info
	split_info = {
		"train_ids": sorted(list(train_ids)),
		"val_ids": sorted(list(val_ids)),
		"test_ids": sorted(list(test_ids))
	}
	try:
		with open(Path(parent_dir) / "data_splits.json", "w") as f: json.dump(split_info, f, indent=4)
	except: pass

	train_paths, train_labels = [], []
	val_paths, val_labels = [], []
	
	for f, l in zip(valid_files, labels):
		if l in train_ids:
			train_paths.append(f)
			train_labels.append(l)
		elif l in val_ids:
			val_paths.append(f)
			val_labels.append(l)

	if logger: logger.info(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

	train_ds = WindowDataset(train_paths, train_labels, cfg, mode="train")
	val_ds = WindowDataset(val_paths, val_labels, cfg, mode="val")

	sampler = BalancedBatchSampler(
		train_labels, 
		batch_size=cfg.batch_size, 
		total_windows=len(train_labels),
		samples_per_class=8
	)

	train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

	return train_loader, val_loader