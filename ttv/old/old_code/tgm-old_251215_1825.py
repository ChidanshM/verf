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
from .models import SiameseFusion, TripletLoss


def get_logger(name: str = "Trainer") -> logging.Logger:
	# Minimal console logger (keeps runner self-contained)
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
	"""
	Tracks validation loss and stops training if it doesn't improve.
	Does NOT save checkpoints (the main loop handles that).
	"""
	def __init__(self, patience: int = 6, min_delta: float = 0.001):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = float("inf")
		self.early_stop = False

	def check(self, val_loss: float) -> bool:
		if val_loss < (self.best_loss - self.min_delta):
			self.best_loss = val_loss
			self.counter = 0  # Reset if we improved
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		return self.early_stop


def main(args):
	logger = get_logger("Trainer")

	# Timestamp at start of run
	timestamp = datetime.now().strftime("%y%m%d_%H%M")
	print(f"Run Timestamp: {timestamp}")

	# Paths
	base_dir = Path(__file__).resolve().parent        # .../verf/ttv
	parent_dir = base_dir.parent                     # .../verf
	data_dir = parent_dir / "processed_tensors"      # .../verf/processed_tensors

	# Device + AMP
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = (device.type == "cuda")
	scaler = GradScaler("cuda", enabled=use_amp)

	logger.info(f"Starting Training on: {device}")

	# Data
	train_loader, val_loader = create_dataloaders(
		data_dir=str(data_dir),
		cfg=CFG,
		parent_dir=str(parent_dir),
		timestamp=timestamp,
		logger=logger,
	)

	# Model / Loss
	model = SiameseFusion(cfg=CFG).to(device)
	criterion = TripletLoss(margin=CFG.margin)

	# Optimiser / Scheduler
	optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode=CFG.sched_mode,
		factor=CFG.sched_factor,
		patience=CFG.sched_patience,
	)

	# Early Stopper (Patience = 6 epochs of no improvement)
	early_stopper = EarlyStopper(patience=6, min_delta=0.0005)

	# Signatures
	model_signature = build_model_signature(CFG)
	training_signature = build_training_signature(
		optimizer_type=optimizer.__class__.__name__,
		scheduler_type=scheduler.__class__.__name__,
	)

	# Save path
	save_path = base_dir / f"best_gait_model-{timestamp}.pth"

	# Resume Logic
	res = load_checkpoint_if_requested(
		resume_arg=args.resume,
		parent_dir=str(parent_dir),
		device=device,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		model_signature=model_signature,
		training_signature=training_signature,
	)

	start_epoch = res.start_epoch
	best_val_loss = res.best_val_loss

	# Sync early stopper with resumed best loss so it doesn't stop immediately
	early_stopper.best_loss = best_val_loss

	if res.resumed:
		logger.info(f"Resumed from: {res.resume_path}")
		logger.info(f"Start epoch: {start_epoch + 1}, Best val loss: {best_val_loss:.4f}")
	else:
		logger.info("No resume (starting fresh).")

	logger.info(f"Saving best checkpoint to: {save_path}")
	logger.info(f"Beginning {CFG.epochs} epochs...")

	# Training loop
	for epoch in range(start_epoch, CFG.epochs):
		loop_start = time.time()

		# ---- Train ----
		model.train()
		train_loss = 0.0

		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
			anchor, pos, neg = batch

			for k in anchor.keys():
				anchor[k] = anchor[k].to(device, non_blocking=True)
				pos[k] = pos[k].to(device, non_blocking=True)
				neg[k] = neg[k].to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)

			with autocast(device_type=device.type, enabled=use_amp):
				emb_a, emb_p, emb_n = model(anchor, pos, neg)
				loss = criterion(emb_a, emb_p, emb_n)

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
				anchor, pos, neg = batch
				for k in anchor.keys():
					anchor[k] = anchor[k].to(device)
					pos[k] = pos[k].to(device)
					neg[k] = neg[k].to(device)

				with autocast(device_type=device.type, enabled=use_amp):
					ea, ep, en = model(anchor, pos, neg)
					loss = criterion(ea, ep, en)

				val_loss += loss.item()

		avg_val_loss = val_loss / max(1, len(val_loader))
		epoch_time = time.time() - loop_start
		current_lr = optimizer.param_groups[0]["lr"]
		current_wd = optimizer.param_groups[0]["weight_decay"]

		logger.info(
			f"Epoch [{epoch+1}/{CFG.epochs}] "
			f"Train: {avg_train_loss:.4f} | Val: val_loss {avg_val_loss:.4f} | {epoch_time:.1f}s | current_lr: {current_lr} | current_wd: {current_wd}"
		)

		# Scheduler Step
		scheduler.step(avg_val_loss)

		# ---- 1. Save Best Model ----
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			ckpt = {
				"epoch": epoch + 1,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"best_val_loss": best_val_loss,
				"model_signature": model_signature,
				"training_signature": training_signature,
			}
			torch.save(ckpt, str(save_path))
			logger.info(f"--> New Best Model Saved! (Loss: {best_val_loss:.4f})")
		
		# ---- 2. Check "Empty Tank" Overfitting (Your Request) ----
		# If train is super low but val is not improving, stop early.
		if avg_train_loss < 0.01 and avg_val_loss > (best_val_loss + 0.05):
			logger.warning(f"Stopping: Train loss {avg_train_loss:.4f} is < 0.01 but Val is lagging.")
			break

		# ---- 3. Standard Early Stopping ----
		if early_stopper.check(avg_val_loss):
			logger.info(f"Early stopping triggered. No validation improvement for {early_stopper.patience} epochs.")
			break

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