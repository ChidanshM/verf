import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- 1. SETUP PATHS (CRITICAL CHANGE) ---
# Add the parent folder ('verf') to Python path so we can import 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Utils
try:
	from utils.loggers import get_logger
	# Logs will save to verf/logs/training.log
	logger = get_logger("Trainer", "training.log")
except ImportError:
	import logging
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger("Trainer")

# --- 2. CONFIGURATION (ADJUSTED) ---
# Data is in verf/processed_tensors, but we are in verf/train
# So we go ".." (up one level) -> "processed_tensors"
DATA_DIR = os.path.join(parent_dir, "processed_tensors")
NUMBER_OF_LIFE=42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
MARGIN = 0.2
INPUT_CHANNELS = 6
WINDOW_SIZE = 200
STRIDE = 50

# --- 3. DATASET CLASS (Same as before) ---
class SixStreamGaitDataset(Dataset):
	# ... [Copy the exact same Dataset code from previous answer] ...
	# (No changes needed here)
	def __init__(self, subjects_data, window_size=200, stride=50, mode='train'):
		self.window_size = window_size
		self.mode = mode
		self.samples = [] 
		self.data = subjects_data
		self.subject_ids = list(subjects_data.keys())
		
		for subj_id in self.subject_ids:
			n_points = len(subjects_data[subj_id]['Pelvis'])
			for start in range(0, n_points - window_size, stride):
				self.samples.append((subj_id, start))

	def __len__(self):
		return len(self.samples)

	def _get_window(self, subj_id, start_idx):
		window_dict = {}
		sensors = ['Pelvis', 'Upper_Spine', 'Shank_LT', 'Foot_LT', 'Shank_RT', 'Foot_RT']
		for sensor in sensors:
			full_signal = self.data[subj_id][sensor]
			signal_slice = full_signal[start_idx : start_idx + self.window_size]
			tensor = signal_slice.clone().detach().float().permute(1, 0)
			window_dict[sensor] = tensor
		return window_dict

	def __getitem__(self, index):
		anchor_subj, anchor_start = self.samples[index]
		anchor_dict = self._get_window(anchor_subj, anchor_start)
		
		if self.mode == 'test':
			return anchor_dict, anchor_subj

		subj_len = len(self.data[anchor_subj]['Pelvis'])
		if subj_len > self.window_size:
			pos_start = np.random.randint(0, subj_len - self.window_size)
		else:
			pos_start = anchor_start
		pos_dict = self._get_window(anchor_subj, pos_start)
		
		other_subjs = [s for s in self.subject_ids if s != anchor_subj]
		if not other_subjs:
			neg_subj = anchor_subj
		else:
			neg_subj = np.random.choice(other_subjs)
			
		neg_len = len(self.data[neg_subj]['Pelvis'])
		if neg_len > self.window_size:
			neg_start = np.random.randint(0, neg_len - self.window_size)
		else:
			neg_start = 0
		neg_dict = self._get_window(neg_subj, neg_start)
		
		return anchor_dict, pos_dict, neg_dict

# --- 4. DATA SPLITTER (Same as before) ---
def create_dataloaders(data_dir):
	logger.info(f"Loading .pt files from: {data_dir}")
	files = glob.glob(os.path.join(data_dir, "*.pt"))
	
	if not files:
		# Critical Error Check
		logger.error(f"No .pt files found in {data_dir}!")
		raise FileNotFoundError(f"Check your path. Current target: {os.path.abspath(data_dir)}")

	np.random.seed(NUMBER_OF_LIFE)
	np.random.shuffle(files)
	
	master_data = {}
	for f in tqdm(files, desc="Loading Data"):
		sub_id = os.path.basename(f).split('.')[0]
		master_data[sub_id] = torch.load(f)

	subject_ids = list(master_data.keys())
	n_total = len(subject_ids)
	n_train = int(n_total * 0.70)
	n_val = int(n_total * 0.15)
	
	train_ids = subject_ids[:n_train]
	val_ids = subject_ids[n_train : n_train + n_val]
	
	def subset_data(ids):
		return {k: master_data[k] for k in ids}

	logger.info(f"Split: Train={len(train_ids)}, Val={len(val_ids)}")
	
	train_ds = SixStreamGaitDataset(subset_data(train_ids), window_size=WINDOW_SIZE, stride=STRIDE, mode='train')
	val_ds = SixStreamGaitDataset(subset_data(val_ids), window_size=WINDOW_SIZE, stride=WINDOW_SIZE, mode='train')
	
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
	
	return train_loader, val_loader

# --- 5. MODEL ARCHITECTURE (Same as before) ---
class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=3, padding=1),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1)
		)
	def forward(self, x):
		return self.cnn(x).squeeze(-1)

class SixStreamFusionNet(nn.Module):
	def __init__(self):
		super(SixStreamFusionNet, self).__init__()
		self.branches = nn.ModuleDict({
			'Pelvis': FeatureExtractor(),
			'Upper_Spine': FeatureExtractor(),
			'Shank_LT': FeatureExtractor(),
			'Foot_LT': FeatureExtractor(),
			'Shank_RT': FeatureExtractor(),
			'Foot_RT': FeatureExtractor()
		})
		self.fusion = nn.Sequential(
			nn.Linear(384, 128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 64)
		)
	def forward(self, inputs):
		feats = [self.branches[key](inputs[key]) for key in inputs.keys()]
		combined = torch.cat(feats, dim=1)
		return self.fusion(combined)

class SiameseFusion(nn.Module):
	def __init__(self):
		super(SiameseFusion, self).__init__()
		self.backbone = SixStreamFusionNet()
	def forward(self, a, p, n):
		return self.backbone(a), self.backbone(p), self.backbone(n)

class TripletLoss(nn.Module):
	def __init__(self, margin=0.2):
		super(TripletLoss, self).__init__()
		self.margin = margin
	def forward(self, anchor, positive, negative):
		dist_pos = torch.pow(anchor - positive, 2).sum(dim=1)
		dist_neg = torch.pow(anchor - negative, 2).sum(dim=1)
		losses = torch.relu(dist_pos - dist_neg + self.margin)
		return losses.mean()

# --- 6. MAIN EXECUTION ---
def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(f"Starting Training on: {device}")
	
	train_loader, val_loader = create_dataloaders(DATA_DIR)
	
	model = SiameseFusion().to(device)
	criterion = TripletLoss(margin=MARGIN)
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
	#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
	scaler = GradScaler()

	best_val_loss = float('inf')
	
	# Save path relative to this script
	save_path = os.path.join(parent_dir, "best_gait_model.pth")

	logger.info(f"Beginning {EPOCHS} epochs...")
	
	for epoch in range(EPOCHS):
		start_time = time.time()
		
		# Train
		model.train()
		train_loss = 0.0
		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
			anchor, pos, neg = batch
			for k in anchor.keys():
				anchor[k] = anchor[k].to(device, non_blocking=True)
				pos[k] = pos[k].to(device, non_blocking=True)
				neg[k] = neg[k].to(device, non_blocking=True)
			
			optimizer.zero_grad()
			with autocast():
				emb_a, emb_p, emb_n = model(anchor, pos, neg)
				loss = criterion(emb_a, emb_p, emb_n)
			
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			train_loss += loss.item()
			
		avg_train_loss = train_loss / len(train_loader)
		
		# Validate
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for batch in val_loader:
				anchor, pos, neg = batch
				for k in anchor.keys():
					anchor[k] = anchor[k].to(device)
					pos[k] = pos[k].to(device)
					neg[k] = neg[k].to(device)
				
				with autocast():
					ea, ep, en = model(anchor, pos, neg)
					loss = criterion(ea, ep, en)
				val_loss += loss.item()
		
		avg_val_loss = val_loss / len(val_loader)
		epoch_time = time.time() - start_time		
		
		logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {epoch_time:.1f}s")
		scheduler.step(avg_val_loss)

		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			torch.save(model.state_dict(), save_path)
			logger.info("--> New Best Model Saved!")

	logger.info("Training Complete.")

if __name__ == "__main__":
	main()