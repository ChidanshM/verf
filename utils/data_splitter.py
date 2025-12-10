import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import your existing Dataset Class
from dataset_loader import SixStreamGaitDataset  # Ensure this file exists

def create_dataloaders(data_dir, batch_size=32, split_ratio=(0.7, 0.15, 0.15)):
	"""
	Splits the .pt files into Train, Val, and Test dataloaders by SUBJECT ID.
	"""
	# 1. List all Subject Files (S001.pt, S002.pt...)
	files = glob.glob(os.path.join(data_dir, "*.pt"))
	subject_ids = [os.path.basename(f).split('.')[0] for f in files]
	
	# Shuffle to ensure random distribution
	# Set seed for reproducibility (Critical for scientific validity)
	np.random.seed(42) 
	np.random.shuffle(files)
	
	n_files = len(files)
	n_train = int(n_files * split_ratio[0])
	n_val = int(n_files * split_ratio[1])
	
	# 2. Split Files
	train_files = files[:n_train]
	val_files = files[n_train : n_train + n_val]
	test_files = files[n_train + n_val:]
	
	print(f"Total Subjects: {n_files}")
	print(f"Training:   {len(train_files)} subjects")
	print(f"Validation: {len(val_files)} subjects")
	print(f"Testing:    {len(test_files)} subjects")
	
	# 3. Helper to Load Data into Memory
	def load_files_to_dict(file_list):
		data_dict = {}
		for fpath in file_list:
			sub_id = os.path.basename(fpath).split('.')[0]
			# Load the dictionary of tensors we saved earlier
			data_dict[sub_id] = torch.load(fpath)
		return data_dict

	# 4. Create Datasets
	print("Loading Training Data...")
	train_data = load_files_to_dict(train_files)
	train_dataset = SixStreamGaitDataset(train_data, window_size=200, stride=50)
	
	print("Loading Validation Data...")
	val_data = load_files_to_dict(val_files)
	# Validation stride can be larger to save time
	val_dataset = SixStreamGaitDataset(val_data, window_size=200, stride=200) 
	
	# Note: Test data isn't returned as a standard DataLoader usually, 
	# because we need custom "Enrollment vs Probe" logic for it.
	# We return the raw dictionary for the Test set.
	test_data = load_files_to_dict(test_files)

	# 5. Create Loaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
							  num_workers=4, pin_memory=True)
	
	# For validation, we might want a simple loader or raw access depending on metric
	# Here we return a loader for calculating Loss
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	
	return train_loader, val_loader, test_data