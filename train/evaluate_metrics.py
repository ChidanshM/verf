import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train.train_gait_model import SiameseFusion, SixStreamGaitDataset

# --- CONFIGURATION ---
DATA_DIR = os.path.join(parent_dir, "processed_tensors")
MODEL_PATH = os.path.join(parent_dir, "best_gait_model.pth")
BATCH_SIZE = 64
WINDOW_SIZE = 200
BLOCK_SIZE = 1000  # Number of rows to process at once (Lowers VRAM usage)

def compute_eer(y_true, y_scores):
	fpr, tpr, thresholds = roc_curve(y_true, y_scores)
	try:
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		thresh = interp1d(fpr, thresholds)(eer)
	except ValueError:
		eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
		thresh = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
	return eer, thresh, fpr, tpr

def evaluate():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Evaluating on: {device}")
	
	# 1. Load Data
	files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
	np.random.seed(42)
	np.random.shuffle(files)
	
	n_total = len(files)
	n_train = int(n_total * 0.70)
	n_val = int(n_total * 0.15)
	test_files = files[n_train + n_val:] 
	
	print(f"Loading {len(test_files)} Test Subjects...")
	test_data = {}
	for f in test_files:
		sub_id = os.path.basename(f).split('.')[0]
		test_data[sub_id] = torch.load(f)
		
	# 2. Load Model
	model = SiameseFusion().to(device)
	model.load_state_dict(torch.load(MODEL_PATH))
	model.eval()
	
	# 3. Generate Embeddings
	print("Generating Embeddings for Test Set...")
	ds = SixStreamGaitDataset(test_data, window_size=WINDOW_SIZE, stride=WINDOW_SIZE, mode='test')
	loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
	
	embeddings = []
	labels = []
	
	with torch.no_grad():
		for anchor_dict, anchor_subjs in tqdm(loader):
			for k in anchor_dict.keys():
				anchor_dict[k] = anchor_dict[k].to(device)
			emb = model.backbone(anchor_dict)
			embeddings.append(emb.cpu())
			labels.extend(anchor_subjs)
			
	embeddings = torch.cat(embeddings) 
	labels = np.array(labels)
	N = len(embeddings)
	
	print(f"Computed {N} embeddings. Calculating distances in blocks...")
	
	# 4. Batched Distance Calculation (Fixes OOM Error)
	genuine_scores = []
	imposter_scores = []
	
	# Move full embeddings to GPU once (if it fits, 40k x 64 floats is tiny ~10MB)
	# The matrix multiplication was the big memory hog, not the embeddings themselves.
	embeddings = embeddings.to(device)
	
	# Pre-calculate norms for ||a-b||^2 = a^2 + b^2 - 2ab
	norms = (embeddings ** 2).sum(dim=1) # (N,)
	
	# Iterate in blocks
	for i in tqdm(range(0, N, BLOCK_SIZE), desc="Processing Blocks"):
		end = min(i + BLOCK_SIZE, N)
		
		# Current batch: (B, 64)
		batch_emb = embeddings[i:end]
		
		# Compute Distances: (B, N)
		# 1. Dot Product
		dot = torch.mm(batch_emb, embeddings.t())
		
		# 2. Euclidean Distance Squared
		# shape broadcasting: (B, 1) + (1, N) - (B, N)
		dist_sq = norms[i:end].unsqueeze(1) + norms.unsqueeze(0) - 2.0 * dot
		dist_chunk = torch.sqrt(torch.relu(dist_sq)).cpu().numpy() # Move to CPU immediately
		
		# 3. Extract Scores (Upper Triangle Only to avoid duplicates)
		batch_labels = labels[i:end]
		
		# Mask: True if same subject
		matches = (batch_labels[:, None] == labels[None, :])
		
		# Loop through rows in this batch to extract valid pairs
		# (We only take columns > row_index to assume upper triangle)
		for row_idx_local in range(len(batch_emb)):
			row_idx_global = i + row_idx_local
			
			# Start from the next element to avoid self-match (dist=0) and duplicates
			if row_idx_global + 1 >= N:
				break
				
			# Slice only the upper triangle part
			valid_slice = slice(row_idx_global + 1, N)
			
			dists = dist_chunk[row_idx_local, valid_slice]
			is_match = matches[row_idx_local, valid_slice]
			
			# Append Genuine
			genuine_scores.extend(dists[is_match])
			
			# Append Imposter (with aggressive downsampling to save RAM)
			imposters = dists[~is_match]
			if len(imposters) > 0:
				# Keep only 5% of imposters to prevent System RAM OOM
				# We still get millions of samples, so it's statistically valid
				if np.random.rand() < 0.05:
					imposter_scores.extend(imposters)
		
		# Cleanup GPU
		del dist_chunk, dist_sq, dot
	
	genuine_scores = np.array(genuine_scores)
	imposter_scores = np.array(imposter_scores)
	
	print(f"\nFinal Pairs -> Genuine: {len(genuine_scores)}, Imposter: {len(imposter_scores)}")
	
	# 5. Compute EER
	y_scores = np.concatenate([genuine_scores, imposter_scores])
	y_true = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(imposter_scores))])
	
	eer, thresh, fpr, tpr = compute_eer(y_true, y_scores)
	
	print("\n" + "="*40)
	print(f"FINAL RESULTS")
	print("="*40)
	print(f"EER (Equal Error Rate): {eer*100:.2f}%")
	print(f"Optimal Threshold:      {thresh:.4f}")
	print("="*40)
	
	# 6. Plot
	plt.figure()
	plt.plot(fpr, tpr, label=f'ROC (EER={eer*100:.2f}%)')
	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.scatter([eer], [1-eer], color='red')
	plt.xlabel('False Acceptance Rate (FAR)')
	plt.ylabel('True Acceptance Rate (TAR)')
	plt.title('Biometric Performance')
	plt.legend()
	save_file = os.path.join(parent_dir, "roc_curve.png")
	plt.savefig(save_file)
	print(f"ROC Curve saved to '{save_file}'")

if __name__ == "__main__":
	evaluate()