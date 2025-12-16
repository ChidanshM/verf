import torch
import numpy as np
import random
import re
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler 
from tqdm import tqdm

from .config import Config

class GaitDataset(Dataset):
	def __init__(self, data_cache, cfg: Config, labels: list, mode: str = "train"):
		self.data_cache = data_cache # Dictionary of loaded tensors
		self.cfg = cfg
		self.labels = labels
		self.mode = mode 
		self.ids = list(data_cache.keys()) # List of IDs for indexing if needed

	def __len__(self):
		# We define length as the number of available labels/files
		return len(self.labels)

	def __getitem__(self, idx):
		# 1. Get Data from RAM (Fast, No Disk I/O)
		# Note: self.labels matches the order of keys passed to init
		# We need a way to map idx -> dictionary key.
		# Let's simplify: passed 'labels' corresponds to sorted keys of data_cache.
		
		# Actually, to be safe, let's store (key, label) pairs in a list
		key, label = self.labels[idx]
		
		full_data = self.data_cache[key]
		
		# 2. Window Slicing (Instant)
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

		# 3. Augmentation
		if self.mode == "train" and random.random() > 0.5:
			for sensor in sliced_data.keys():
				sliced_data[sensor] = -sliced_data[sensor]

		return sliced_data, label

class BalancedBatchSampler(Sampler):
	def __init__(self, labels, batch_size, samples_per_class=8):
		# labels here is just the list of ints [1, 1, 2, 2, ...]
		self.labels = labels
		self.batch_size = batch_size
		self.samples_per_class = samples_per_class
		self.classes_per_batch = self.batch_size // self.samples_per_class
		
		self.label_to_indices = defaultdict(list)
		for idx, label in enumerate(labels):
			self.label_to_indices[label].append(idx)
			
		self.unique_labels = list(self.label_to_indices.keys())
		# Restore ~15k runs feel: Visit every user ~50 times per epoch
		self.n_batches = int(len(self.unique_labels) // self.classes_per_batch) * 50

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

	# --- 1. LOAD EVERYTHING TO RAM (Like Old Code) ---
	if logger: logger.info("Loading all data into RAM...")
	master_cache = {}
	valid_keys = []
	all_labels = []

	for f in tqdm(all_files, desc="Loading Dataset"):
		# Parse Label
		stem = f.name.split('_')[0].split('.')[0]
		numeric_part = re.sub(r'\D', '', stem)
		if not numeric_part: continue
		
		label = int(numeric_part)
		
		# Load & Cache
		# Map location cpu ensures it sits in System RAM, not GPU
		data = torch.load(f, map_location='cpu') 
		
		key = f.name # Unique ID for dictionary
		master_cache[key] = data
		
		valid_keys.append(key)
		all_labels.append(label)

	# --- 2. SPLIT ---
	# We split IDs, not files (Strict subject separation)
	unique_ids = sorted(list(set(all_labels)))
	n_ids = len(unique_ids)
	idx_train = int(n_ids * 0.70)
	idx_val = int(n_ids * 0.85)
	
	train_ids = set(unique_ids[:idx_train])
	val_ids = set(unique_ids[idx_train:idx_val])
	
	# Prepare lists for Dataset: [(key, label), (key, label)...]
	train_items = [] 
	val_items = []
	
	# Also separate labels list for the Sampler
	train_labels_for_sampler = []
	
	for key, label in zip(valid_keys, all_labels):
		if label in train_ids:
			train_items.append((key, label))
			train_labels_for_sampler.append(label)
		elif label in val_ids:
			val_items.append((key, label))

	if logger:
		logger.info(f"Split :: Train: {len(train_ids)} IDs | Val: {len(val_ids)} IDs")
		logger.info(f"RAM Cache :: {len(master_cache)} total files loaded.")

	# --- 3. DATASETS ---
	# Pass the HUGE master_cache to both. They will only access their specific keys.
	train_ds = GaitDataset(master_cache, cfg, train_items, mode="train")
	val_ds = GaitDataset(master_cache, cfg, val_items, mode="val")

	# --- 4. SAMPLER & LOADERS ---
	sampler = BalancedBatchSampler(train_labels_for_sampler, batch_size=cfg.batch_size, samples_per_class=8)

	# Since data is in RAM, num_workers=0 is usually fastest (no pickling overhead).
	# If you want background processing, you can try 2, but 0 is safe.
	train_loader = DataLoader(
		train_ds, 
		batch_sampler=sampler, 
		num_workers=0, 
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_ds, 
		batch_size=cfg.batch_size, 
		shuffle=False, 
		num_workers=0
	)

	return train_loader, val_loader