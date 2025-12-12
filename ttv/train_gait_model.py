import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path
from dataclasses import dataclass


timestamp = datetime.now().strftime("%y%m%d_%H%M")
print(timestamp)

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


@dataclass(frozen=True)
class Config:
    seed: int = 42 #NUMBER_OF_LIFE
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    margin: float = 0.2
    input_channels: int = 6
    window_size: int = 200
    stride: int = 50
    streams: tuple = ("Pelvis","Upper_Spine","Shank_LT","Foot_LT","Shank_RT","Foot_RT")

CFG = Config()

MODEL_SIGNATURE = {
	"model_name": "SiameseFusion/SixStreamFusionNet",
	"input_channels": CFG.input_channels,
	"window_size": CFG.window_size,
	"streams": list(CFG.streams),
	"embedding_dim": 64,
	"fusion_in": 64 * len(CFG.streams),
}

TRAINING_SIGNATURE = {
	"optimizer_type": "AdamW",
	"scheduler_type": "ReduceLROnPlateau",
}


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

	np.random.seed(CFG.seed)
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
	
	# 1. Define the Test IDs (The remaining 15%)
	test_ids = subject_ids[n_train + n_val:]

	# 2. Save them to a file so evaluation.ipynb uses the EXACT same people
	test_ids_path = os.path.join(parent_dir, "test_subjects.txt")
	with open(test_ids_path, "w") as f:
		for sid in test_ids:
			f.write(sid + "\n")

	logger.info(f"Saved {len(test_ids)} test subjects to {test_ids_path}")
	
	def subset_data(ids):
		return {k: master_data[k] for k in ids}

	logger.info(f"Split: Train={len(train_ids)}, Val={len(val_ids)}")
	
	train_ds = SixStreamGaitDataset(subset_data(train_ids), cfg=CFG, mode="train")
	val_ds   = SixStreamGaitDataset(subset_data(val_ids),   cfg=CFG, mode="train")

	
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
	
	return train_loader, val_loader

# --- 5. MODEL ARCHITECTURE (Same as before) ---
class FeatureExtractor(nn.Module):
class FeatureExtractor(nn.Module):
	def __init__(self, input_channels: int, out_channels1=32, out_channels2=64):
		super().__init__()
		self.cnn = nn.Sequential(
			nn.Conv1d(input_channels, out_channels1, kernel_size=3, padding=1),
			nn.BatchNorm1d(out_channels1),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(out_channels1, out_channels2, kernel_size=3, padding=1),
			nn.BatchNorm1d(out_channels2),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1),
		)
	def forward(self, x):
		return self.cnn(x).squeeze(-1)

class SixStreamFusionNet(nn.Module):
class SixStreamGaitDataset(Dataset):
	def __init__(self, subjects_data, cfg: Config, mode="train"):
		super(SixStreamFusionNet, self).__init__()
		self.cfg = cfg
		self.window_size = cfg.window_size
		self.mode = mode
		self.samples = []
		self.data = subjects_data
		self.subject_ids = list(subjects_data.keys())
		self.sensors = list(cfg.streams)
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
	
def _find_checkpoints(parent_dir: str):
	p = Path(parent_dir)
	return sorted(p.glob("best_gait_model-*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)

def _resolve_resume_path(parent_dir: str, resume: str | None):
	"""
	resume:
	  - None      => prompt if checkpoints exist
	  - "latest"  => pick newest checkpoint
	  - "<path>"  => use that file (relative paths resolved from parent_dir)
	Returns Path or None.
	"""
	checkpoints = _find_checkpoints(parent_dir)

	if resume is None or resume == "prompt":
		if not checkpoints:
			return None

		print("\nFound existing checkpoints:")
		for i, ck in enumerate(checkpoints[:10], start=1):
			print(f"  [{i}] {ck.name}")
		print("  [0] Start fresh\n")

		try:
			choice = input("Resume from which checkpoint? (number, or 0): ").strip()
		except EOFError:
			return None

		if not choice.isdigit():
			return None

		idx = int(choice)
		if idx == 0:
			return None
		if 1 <= idx <= min(len(checkpoints), 10):
			return checkpoints[idx - 1]
		return None

	if resume.lower() == "latest":
		return checkpoints[0] if checkpoints else None

	rp = Path(resume)
	if not rp.is_absolute():
		rp = Path(parent_dir) / rp
	return rp if rp.exists() else None


# --- 6. MAIN EXECUTION ---
def main(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(f"Starting Training on: {device}")
	
	train_loader, val_loader = create_dataloaders(DATA_DIR)
	
	model = SiameseFusion().to(device)
	criterion = TripletLoss(margin=MARGIN)
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
	
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)#), verbose=True)
	use_amp = (device.type == "cuda")
	scaler = GradScaler("cuda", enabled=use_amp)

	best_val_loss = float('inf')
	
	# Save path relative to this script
	save_path = os.path.join(parent_dir, f"best_gait_model-{timestamp}.pth")

	start_epoch = 0
	best_val_loss = float("inf")

	# Only attempt resume if user explicitly asked
	if args.resume is not None:
		resume_path = _resolve_resume_path(parent_dir, args.resume)  # prompt/latest/path
		if resume_path is None:
			raise RuntimeError("Resume requested but no valid checkpoint selected/found.")
		checkpoint = torch.load(str(resume_path), map_location=device)

		# 1) Model architecture must match
		if checkpoint.get("model_signature") != MODEL_SIGNATURE:
			raise RuntimeError("Incompatible checkpoint: model architecture/signature mismatch.")

		# 2) Optimiser type must match
		ck_tr = checkpoint.get("training_signature", {})
		if ck_tr.get("optimizer_type") != TRAINING_SIGNATURE["optimizer_type"]:
			raise RuntimeError("Incompatible checkpoint: optimiser type mismatch.")

		# 3) Scheduler type must match
		if ck_tr.get("scheduler_type") != TRAINING_SIGNATURE["scheduler_type"]:
			raise RuntimeError("Incompatible checkpoint: scheduler type mismatch.")

		# Load everything only if all match
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		if "scheduler_state_dict" in checkpoint:
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

		start_epoch = checkpoint.get("epoch", 0)
		best_val_loss = checkpoint.get("best_val_loss", float("inf"))

		logger.info(f"Resumed from {resume_path} at epoch {start_epoch+1}, best_val_loss={best_val_loss:.4f}")
	else:
		logger.info("Starting fresh (no --resume provided).")

	
	for epoch in range(start_epoch,EPOCHS+start_epoch):
		start_time = time.time()
		
		# Train
		model.train()
		train_loss = 0.0
		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, mininterval=1.0):
			anchor, pos, neg = batch
			for k in anchor.keys():
				anchor[k] = anchor[k].to(device, non_blocking=True)
				pos[k] = pos[k].to(device, non_blocking=True)
				neg[k] = neg[k].to(device, non_blocking=True)
			
			optimizer.zero_grad()
			with autocast(device_type=device.type, enabled=use_amp):
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
				
				with autocast(device_type=device.type, enabled=use_amp):
					ea, ep, en = model(anchor, pos, neg)
					loss = criterion(ea, ep, en)
				val_loss += loss.item()
		
		avg_val_loss = val_loss / len(val_loader)
		epoch_time = time.time() - start_time		
		
		logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {epoch_time:.1f}s")
		scheduler.step(avg_val_loss)

		# [REPLACE THE SAVING BLOCK WITH THIS]
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			
			# Create a comprehensive checkpoint dictionary
			
			checkpoint = {
				"epoch": epoch + 1,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"best_val_loss": best_val_loss,
				"model_signature": MODEL_SIGNATURE,
				"training_signature": TRAINING_SIGNATURE,
			}

			torch.save(checkpoint, save_path)
			logger.info(f"--> New Best Model Saved! (Loss: {best_val_loss:.4f})")
		
		# if avg_val_loss < best_val_loss:
		# 	best_val_loss = avg_val_loss
		# 	torch.save(model.state_dict(), save_path)
		# 	logger.info("--> New Best Model Saved!")

	logger.info("Training Complete.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--resume",
		nargs="?",
		const="prompt",
		default=None,
		help='Resume training. Use "--resume" to prompt, "--resume latest", or "--resume <path/to/.pth>".',
	)

	args = parser.parse_args()
	main(args)
