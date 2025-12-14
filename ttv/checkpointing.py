from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class ResumeResult:
	resumed: bool
	resume_path: Optional[Path] = None
	start_epoch: int = 0
	best_val_loss: float = float("inf")


def find_checkpoints(parent_dir: str | Path, pattern: str = "best_gait_model-*.pth") -> list[Path]:
	"""Return checkpoints sorted by modified time (newest first)."""
	p = Path(parent_dir)
	return sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)


def resolve_resume_path(parent_dir: str | Path, resume_arg: Optional[str]) -> Optional[Path]:
	"""
	Per your requirement: don't look/prompt unless resume_arg is provided.

	resume_arg meanings:
	  - None      -> do not resume, do not scan
	  - "prompt"  -> list & prompt (top 10)
	  - "latest"  -> newest checkpoint
	  - "<path>"  -> use that path (relative resolved from parent_dir)
	"""
	if resume_arg is None:
		return None  # ✅ do not scan

	parent_dir = Path(parent_dir)

	if resume_arg == "prompt":
		ckpts = find_checkpoints(parent_dir)
		if not ckpts:
			return None

		print("\nFound existing checkpoints:")
		for i, ck in enumerate(ckpts[:10], start=1):
			print(f"  [{i}] {ck.name}")
		print("  [0] Cancel (start fresh)\n")

		try:
			choice = input("Resume from which checkpoint? (number, or 0): ").strip()
		except EOFError:
			return None

		if not choice.isdigit():
			return None

		idx = int(choice)
		if idx == 0:
			return None
		if 1 <= idx <= min(len(ckpts), 10):
			return ckpts[idx - 1]
		return None

	if resume_arg.lower() == "latest":
		ckpts = find_checkpoints(parent_dir)
		return ckpts[0] if ckpts else None

	rp = Path(resume_arg)
	if not rp.is_absolute():
		rp = parent_dir / rp
	return rp if rp.exists() else None


def _require_equal(label: str, got: Any, expected: Any) -> None:
	if got != expected:
		raise RuntimeError(
			f"Incompatible checkpoint: {label} mismatch.\n"
			f"Checkpoint {label}: {got}\n"
			f"Current {label}:   {expected}"
		)


def load_checkpoint_if_requested(
	*,
	resume_arg: Optional[str],
	parent_dir: str | Path,
	device: torch.device,
	model,
	optimizer,
	scheduler,
	model_signature: Dict[str, Any],
	training_signature: Dict[str, Any],
) -> ResumeResult:
	"""
	Loads checkpoint ONLY if resume_arg is provided.

	Enforces:
	  - model_signature exact match
	  - training_signature exact match (in your config it's optimiser type + scheduler type only)
	"""
	resume_path = resolve_resume_path(parent_dir, resume_arg)

	# If user explicitly asked to resume but none selected/found -> hard fail
	if resume_arg is not None and resume_path is None:
		raise RuntimeError("Resume requested, but no valid checkpoint was selected/found.")

	if resume_path is None:
		return ResumeResult(resumed=False)

	ckpt = torch.load(str(resume_path), map_location=device)

	_require_equal("model_signature", ckpt.get("model_signature"), model_signature)
	_require_equal("training_signature", ckpt.get("training_signature"), training_signature)

	model.load_state_dict(ckpt["model_state_dict"])
	optimizer.load_state_dict(ckpt["optimizer_state_dict"])
	if "scheduler_state_dict" in ckpt:
		scheduler.load_state_dict(ckpt["scheduler_state_dict"])

	start_epoch = int(ckpt.get("epoch", 0))
	best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

	return ResumeResult(
		resumed=True,
		resume_path=resume_path,
		start_epoch=start_epoch,
		best_val_loss=best_val_loss,
	)
