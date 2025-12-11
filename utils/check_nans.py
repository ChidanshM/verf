import torch
import glob
import os
from tqdm import tqdm

DATA_DIR = "database/processed_tensors"  # Adjust if needed

def check_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    print(f"Checking {len(files)} files for corruption...")
    
    bad_files = []
    
    for f in tqdm(files):
        try:
            data = torch.load(f)
            for sensor, tensor in data.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"❌ CORRUPTION found in {os.path.basename(f)} ({sensor})")
                    bad_files.append(f)
                    break
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
            bad_files.append(f)
            
    if bad_files:
        print(f"\nFound {len(bad_files)} bad files. You should delete them or re-run prepare_data.")
        # Optional: Delete them automatically
        # for f in bad_files: os.remove(f)
    else:
        print("\n✅ Data is clean. The issue is likely the Training Hyperparameters.")

if __name__ == "__main__":
    check_data()