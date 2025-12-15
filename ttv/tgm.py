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


def main(args):
	logger = get_logger("Trainer")

	# Timestamp at start of run (used for THIS run's save_path and test_subjects file)
	timestamp = datetime.now().strftime("%y%m%d_%H%M")
	print(timestamp)

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

	# Signatures (for strict resume checks)
	model_signature = build_model_signature(CFG)
	training_signature = build_training_signature(
		optimizer_type=optimizer.__class__.__name__,
		scheduler_type=scheduler.__class__.__name__,
	)

	# This run saves to a NEW timestamped file (always)
	save_path = base_dir/ "new" / f"best_gait_model-{timestamp}.pth"

	# Resume ONLY if user explicitly asked via CLI
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

	if res.resumed:
		logger.info(f"Resumed from: {res.resume_path}")
		logger.info(f"Start epoch: {start_epoch + 1}, Best val loss: {best_val_loss:.4f}")
	else:
		logger.info("No resume (starting fresh).")

	logger.info(f"Saving best checkpoint to: {save_path}")
	logger.info(f"Beginning {CFG.epochs} epochs...")

	# Training loop
	for epoch in range(start_epoch, CFG.epochs):
		start_time = time.time()

		# ---- Train ----
		model.train()
		train_loss = 0.0

		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, mininterval=1.0):
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
		epoch_time = time.time() - start_time

		logger.info(
			f"Epoch [{epoch+1}/{CFG.epochs}] "
			f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {epoch_time:.1f}s"
		)

		scheduler.step(avg_val_loss)

		# ---- Save best (with signatures for strict resume) ----
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

	logger.info("Training Complete.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Per your rule: no scanning/prompting unless this flag is present.
	# --resume           => prompts (because const="prompt")
	# --resume latest    => loads newest
	# --resume <path>    => loads that file
	parser.add_argument(
		"--resume",
		nargs="?",
		const="prompt",
		default=None,
		help='Resume training. Use "--resume" to prompt, "--resume latest", or "--resume <path/to/.pth>".',
	)

	args = parser.parse_args()
	main(args)