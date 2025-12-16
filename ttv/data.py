# verf/ttv/data.py
import torch
import numpy as np
import random
import re
import shutil
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

        # --- Z-Score Normalization (Per Window) ---
        # "Normalize inputs before feeding them in"
        for s in data.keys():
            tensor = data[s] # Shape (C, T)
            mean = tensor.mean(dim=1, keepdim=True)
            std = tensor.std(dim=1, keepdim=True) + 1e-6
            data[s] = (tensor - mean) / std

        # Augmentation (Mirroring)
        if self.mode == "train" and random.random() > 0.5:
            for sensor in data.keys():
                data[sensor] = -data[sensor]

        return data, label

    def _get_dummy(self):
        d = {}
        for s in self.cfg.streams:
            d[s] = torch.zeros((self.cfg.input_channels, self.cfg.window_size))
        return d

# ... (Keep BalancedBatchSampler and generate_cache logic same as previous turn) ...
# Just make sure generate_cache uses cfg.window_size (500 now)

# Ensure create_dataloaders uses the correct batch_size
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, samples_per_class=8):
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        # Ensure we don't crash if batch_size is small
        if self.batch_size < self.samples_per_class:
            self.samples_per_class = self.batch_size
        
        self.classes_per_batch = self.batch_size // self.samples_per_class
        
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
            
        self.unique_labels = list(self.label_to_indices.keys())
        
        # Visit every user ~5 times per epoch
        if not self.unique_labels:
            self.n_batches = 0
        else:
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

def generate_cache(data_dir: Path, cache_dir: Path, cfg: Config, logger):
    if logger: logger.info(f"Generating cache at: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_files = sorted(list(data_dir.glob("*.pt")))
    
    stride = cfg.window_size // 2
    count = 0
    
    for f in tqdm(all_files, desc="Caching Windows"):
        stem = f.name.split('_')[0].split('.')[0]
        try:
            full_data = torch.load(f, map_location='cpu')
        except: continue
        
        first_val = next(iter(full_data.values()))
        seq_len = first_val.shape[-1]
        
        for i, start in enumerate(range(0, seq_len - cfg.window_size + 1, stride)):
            window = {}
            for k, v in full_data.items():
                window[k] = v[..., start : start + cfg.window_size].clone()
            
            torch.save(window, cache_dir / f"{stem}_{i:05d}.pt")
            count += 1
        del full_data

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
    source_dir = Path(data_dir)
    cache_dir = source_dir.parent / "cache_windows_transformer" # New cache for new window size
    
    if not cache_dir.exists() or not any(cache_dir.iterdir()):
        generate_cache(source_dir, cache_dir, cfg, logger)
    else:
        if logger: logger.info(f"Using cache: {cache_dir}")

    all_files = sorted(list(cache_dir.glob("*.pt")))
    labels = []
    valid_files = []
    
    for f in all_files:
        stem = f.name.split('_')[0]
        numeric_part = re.sub(r'\D', '', stem)
        if numeric_part:
            labels.append(int(numeric_part))
            valid_files.append(f)

    # Split 70/15/15
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

    if logger: logger.info(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = WindowDataset(train_paths, train_labels, cfg, mode="train")
    val_ds = WindowDataset(val_paths, val_labels, cfg, mode="val")

    sampler = BalancedBatchSampler(train_labels, batch_size=cfg.batch_size, samples_per_class=8)

    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
