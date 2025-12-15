import torch
import numpy as np
import random
import re  # Added regex for safer parsing
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler 

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
        # Flip signals with 50% probability
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
        self.count = 0
        while self.count + self.batch_size <= self.n_samples:
            # 1. Pick P random classes
            classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
            
            indices = []
            for class_ in classes:
                # 2. Pick K samples for this class (allow replacement if class is small)
                class_indices = self.label_to_indices[class_]
                
                # Check if we need replacement (if class has fewer samples than K)
                replace_flag = len(class_indices) < self.samples_per_class
                
                selected = np.random.choice(class_indices, self.samples_per_class, replace=replace_flag)
                indices.extend(selected)
                
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_batches

def create_dataloaders(data_dir: str, cfg: Config, parent_dir: str, timestamp: str, logger):
    # Gather files
    all_files = sorted(list(Path(data_dir).glob("*.pt")))
    if not all_files:
        raise RuntimeError(f"No .pt files found in {data_dir}")

    # Extract Labels ROBUSTLY
    # Handles: "S001.pt", "S001_session1.pt", "001.pt"
    labels = []
    valid_files = []
    
    for f in all_files:
        # Extract the first part of the filename (e.g., "S001")
        stem = f.name.split('_')[0].split('.')[0] # Split by _ or . to get ID
        
        # Remove non-digit characters (like "S")
        numeric_part = re.sub(r'\D', '', stem) 
        
        if numeric_part:
            labels.append(int(numeric_part))
            valid_files.append(f)
        else:
            if logger:
                logger.warning(f"Skipping file with invalid ID: {f.name}")

    # Train/Val Split (80/20 by Subject ID)
    unique_ids = sorted(list(set(labels)))
    split_idx = int(len(unique_ids) * 0.8)
    
    train_ids = set(unique_ids[:split_idx])
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    for f, l in zip(valid_files, labels):
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