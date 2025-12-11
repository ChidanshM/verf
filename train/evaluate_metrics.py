import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
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

def compute_eer(y_true, y_scores):
    """
    y_true: 1 for Imposter, 0 for Genuine (Standard ROC convention)
    y_scores: Distance values (Higher distance = More likely Imposter)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # FRR = 1 - TPR
    frr = 1 - tpr
    
    # EER is where FPR crosses FRR
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
    except ValueError:
        print("Error finding EER intersection. Returning estimate.")
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        thresh = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        
    return eer, thresh, fpr, tpr

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    # 1. Load Data
    # We need the Test Split. We re-run the split logic to ensure we get the same test set.
    # (In production, you would save the 'test_ids' list to a file)
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    np.random.seed(42)
    np.random.shuffle(files)
    
    n_total = len(files)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    test_files = files[n_train + n_val:] # Last 15%
    
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
    # We create a simple dataset to iterate over windows
    ds = SixStreamGaitDataset(test_data, window_size=WINDOW_SIZE, stride=WINDOW_SIZE, mode='test')
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for anchor_dict, anchor_subjs in tqdm(loader):
            # Move to GPU
            for k in anchor_dict.keys():
                anchor_dict[k] = anchor_dict[k].to(device)
            
            # Forward Pass (Backbone only)
            emb = model.backbone(anchor_dict)
            embeddings.append(emb.cpu())
            labels.extend(anchor_subjs)
            
    embeddings = torch.cat(embeddings) # (N, 64)
    labels = np.array(labels)
    
    print(f"Computed {len(embeddings)} embeddings.")
    
    # 4. Calculate Distance Matrix (GPU Accelerated)
    print("Computing Distance Matrix...")
    embeddings = embeddings.to(device)
    
    # Dist Matrix: ||a-b||^2 = a^2 + b^2 - 2ab
    dot_product = torch.mm(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    dist_sq = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    dist_matrix = torch.sqrt(torch.relu(dist_sq)).cpu().numpy() # Move to CPU
    
    # 5. Extract Genuine & Imposter Scores
    genuine_scores = []
    imposter_scores = []
    
    # Create Label Mask (True if same subject)
    label_mat = labels[:, None] == labels[None, :]
    
    # Use upper triangle indices to avoid duplicates and self-comparison
    rows, cols = np.triu_indices(len(labels), k=1)
    
    print("Sorting Scores...")
    # This loop can be slow for large N. Vectorized masking is faster:
    # Vectorized extraction:
    dists = dist_matrix[rows, cols]
    matches = label_mat[rows, cols]
    
    genuine_scores = dists[matches]
    imposter_scores = dists[~matches]
    
    # Downsample imposters if massive (optional)
    if len(imposter_scores) > 1_000_000:
        imposter_scores = np.random.choice(imposter_scores, 1_000_000, replace=False)
        
    print(f"Genuine Pairs: {len(genuine_scores)}")
    print(f"Imposter Pairs: {len(imposter_scores)}")
    
    # 6. Compute EER
    # Labels: 0 = Genuine, 1 = Imposter
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    y_true = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(imposter_scores))])
    
    eer, thresh, fpr, tpr = compute_eer(y_true, y_scores)
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS")
    print("="*40)
    print(f"EER (Equal Error Rate): {eer*100:.2f}%")
    print(f"Optimal Threshold:      {thresh:.4f}")
    print("="*40)
    
    # 7. Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (EER={eer*100:.2f}%)')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.scatter([eer], [1-eer], color='red')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.title('Biometric Performance')
    plt.legend()
    plt.savefig(os.path.join(parent_dir, "roc_curve.png"))
    print("ROC Curve saved to 'verf/roc_curve.png'")

if __name__ == "__main__":
    evaluate()