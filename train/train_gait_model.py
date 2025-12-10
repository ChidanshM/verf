import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
# Define the sensors and columns exactly as they appear in your CSV
SENSORS = {
    "Pelvis":      ["Pelvis-Acceleration", "Pelvis-Gyroscope"],
    "Upper_Spine": ["Upper spine-Acceleration", "Upper spine-Gyroscope"],
    "Shank_LT":    ["Shank LT-Acceleration", "Shank LT-Gyroscope"],
    "Foot_LT":     ["Foot LT-Acceleration", "Foot LT-Gyroscope"],
    "Shank_RT":    ["Shank RT-Acceleration", "Shank RT-Gyroscope"],
    "Foot_RT":     ["Foot RT-Acceleration", "Foot RT-Gyroscope"]
}

# Adjust paths to match your folder structure
DATA_ROOT = "path/to/NONAN_Dataset" 

def butter_highpass_filter(data, cutoff=0.5, fs=200, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def process_single_file(filepath):
    try:
        df = pd.read_csv(filepath)
        file_data = {}
        
        for sensor_name, features in SENSORS.items():
            # Construct column names based on your target_columns.csv format
            # e.g., "Noraxon MyoMotion-Segments-Pelvis-Acceleration-x (mG)"
            acc_cols = [f"Noraxon MyoMotion-Segments-{features[0]}-{axis} (mG)" for axis in ['x','y','z']]
            gyro_cols = [f"Noraxon MyoMotion-Segments-{features[1]}-{axis} (deg/s)" for axis in ['x','y','z']]
            
            # 1. Load Raw
            acc = df[acc_cols].values
            gyro = df[gyro_cols].values
            
            # 2. Gravity Removal (High Pass)
            acc_clean = butter_highpass_filter(acc, cutoff=0.5, fs=200)
            
            # 3. Scaling
            acc_clean = acc_clean / 1000.0  # mG -> G
            gyro_clean = gyro / 500.0       # deg/s -> Normalized
            
            # 4. Stack (6 Channels)
            # If you decided to go Accel-Only, verify this line!
            # Current: Stacking Acc + Gyro = 6 Channels
            file_data[sensor_name] = np.hstack([acc_clean, gyro_clean]).astype(np.float32)
            
        return file_data
        
    except KeyError as e:
        print(f"Skipping {filepath}: Missing column {e}")
        return None

def build_master_dictionary():
    master_dict = {}
    
    # Find all CSV files (assuming folder structure: Root/SubjectID/Trial.csv)
    # You might need to adjust the glob pattern
    search_pattern = os.path.join(DATA_ROOT, "**", "*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} files. Processing...")
    
    for f in files:
        # Extract Subject ID from folder name or filename
        # Example: .../S001/Walk01.csv -> subject_id = "S001"
        subject_id = os.path.basename(os.path.dirname(f)) 
        
        processed_data = process_single_file(f)
        
        if processed_data is not None:
            # If subject already exists, concatenate the new trial (make one long sequence)
            if subject_id in master_dict:
                for sensor in SENSORS.keys():
                    master_dict[subject_id][sensor] = np.concatenate(
                        [master_dict[subject_id][sensor], processed_data[sensor]], axis=0
                    )
            else:
                master_dict[subject_id] = processed_data
                
    print(f"Loaded {len(master_dict)} subjects successfully.")
    return master_dict

if __name__ == "__main__":
    data = build_master_dictionary()
    # Optional: Save it to disk so you don't have to re-process every time
    # import pickle
    # with open('processed_data.pkl', 'wb') as f:
    #     pickle.dump(data, f)