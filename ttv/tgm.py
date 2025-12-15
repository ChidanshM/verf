# verf/ttv/tgm.py
from __future__ import annotations
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .checkpointing import load_checkpoint_if_requested
from .config import CFG, build_model_signature, build_training_signature
from .data import create_dataloaders
from .models import SixStreamFusionNet, BatchHardTripletLoss

def get_logger(name: str = "Trainer") -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		logger.setLevel(logging.INFO)
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
		ch.setFormatter(fmt)
		logger.addHandler(ch)
	return logger

class EarlyStopper:
	def __init__(self, patience: int = 10, min_delta: float = 0.001):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = float("inf")
		self.early_stop = False

	def check(self, val_loss: float) -> bool:
		if val_loss < (self.best_loss - self.min_delta):
			self.best_loss = val_loss
			self.counter = 0 
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		return self.early_stop

def main(args):
	logger = get_logger("Trainer")
	timestamp = datetime.now().strftime("%y%m%d_%H%M")
	
	base_dir = Path(__file__).resolve().parent
	parent_dir = base_dir.parent
	data_dir = parent_dir / "processed_tensors"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = (device.type == "cuda")
	scaler = GradScaler("cuda", enabled=use_amp)

	logger.info(f"Starting Batch Hard Training on: {device}")

	# Data
	train_loader, val_loader = create_dataloaders(
		data_dir=str(data_dir), cfg=CFG, parent_dir=str(parent_dir),
		timestamp=timestamp, logger=logger
	)

	# Model: Direct SixStreamNet (No Siamese Wrapper needed)
	model = SixStreamFusionNet(cfg=CFG).to(device)
	
	# Loss: Batch Hard
	criterion = BatchHardTripletLoss(margin=CFG.margin)

	optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode=CFG.sched_mode, factor=CFG.sched_factor, patience=CFG.sched_patience
	)

	early_stopper = EarlyStopper(patience=10, min_delta=0.0005)
	model_signature = build_model_signature(CFG)
	training_signature = build_training_signature(optimizer.__class__.__name__, scheduler.__class__.__name__)

	save_path = base_dir / f"best_gait_model-{timestamp}.pth"
	
	start_epoch = 0
	best_val_loss = float('inf')

	# (Resume logic omitted for brevity, but compatible if added back)

	logger.info(f"Beginning {CFG.epochs} epochs...")

	for epoch in range(start_epoch, CFG.epochs):
		loop_start = time.time()

		# ---- Train ----
		model.train()
		train_loss = 0.0

		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
			# Unpack: (Inputs, Labels)
			inputs, labels = batch
			labels = labels.to(device)
			for k in inputs.keys():
				inputs[k] = inputs[k].to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)

			with autocast(device_type=device.type, enabled=use_amp):
				# Forward pass: Get embeddings for entire batch
				embeddings = model(inputs)
				# Compute Batch Hard Loss
				loss = criterion(embeddings, labels)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			train_loss += loss.item()

		avg_train_loss = train_loss / max(1, len(train_loader))

		# ---- Validate ----
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for batch in val_loader:
				inputs, labels = batch
				labels = labels.to(device)
				for k in inputs.keys():
					inputs[k] = inputs[k].to(device)

				with autocast(device_type=device.type, enabled=use_amp):
					embeddings = model(inputs)
					loss = criterion(embeddings, labels)
				
				val_loss += loss.item()

		avg_val_loss = val_loss / max(1, len(val_loader))
		epoch_time = time.time() - loop_start

		current_lr = optimizer.param_groups[0]["lr"]
		
		logger.info(
			f"Epoch [{epoch+1}/{CFG.epochs}] "
			f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
			f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
		)

		scheduler.step(avg_val_loss)

		# Save Best
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			ckpt = {
				"epoch": epoch + 1,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"best_val_loss": best_val_loss,
			}
			torch.save(ckpt, str(save_path))
			logger.info(f"--> New Best Model Saved! (Loss: {best_val_loss:.4f})")

		# Early Stop
		if early_stopper.check(avg_val_loss):
			logger.info("Early stopping triggered.")
			break

	logger.info("Training Complete.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--resume", nargs="?", const="prompt", default=None)
	args = parser.parse_args()
	main(args)