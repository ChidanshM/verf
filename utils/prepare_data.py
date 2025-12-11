import os
import glob
import re
import pandas as pd
import numpy as np
import torch
from scipy.signal import butter, filtfilt
import logging
from tqdm import tqdm  # Requires: pip install tqdm

# --- CONFIGURATION ---
INPUT_FOLDER = os.path.join( "DATA", "fep","er") 
OUTPUT_FOLDER = os.path.join( "database","processed_tensors") 
LOG_FILE = "data_preparation.log"

# Setup Logging (File + Console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

SENSORS_MAP = {
    "Pelvis": {
        "acc": ["Noraxon MyoMotion-Segments-Pelvis-Acceleration-x (mG)", 
                "Noraxon MyoMotion-Segments-Pelvis-Acceleration-y (mG)", 
                "Noraxon MyoMotion-Segments-Pelvis-Acceleration-z (mG)"],
        "gyro": ["Noraxon MyoMotion-Segments-Pelvis-Gyroscope-x (deg/s)", 
                 "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-y (deg/s)", 
                 "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-z (deg/s)"]
    },
    "Upper_Spine": {
        "acc": ["Noraxon MyoMotion-Segments-Upper spine-Acceleration-x (mG)", 
                "Noraxon MyoMotion-Segments-Upper spine-Acceleration-y (mG)", 
                "Noraxon MyoMotion-Segments-Upper spine-Acceleration-z (mG)"],
        "gyro": ["Noraxon MyoMotion-Segments-Upper spine-Gyroscope-x (deg/s)", 
                 "Noraxon MyoMotion-Segments-Upper spine-Gyroscope-y (deg/s)", 
                 "Noraxon MyoMotion-Segments-Upper spine-Gyroscope-z (deg/s)"]
    },
    "Shank_LT": {
        "acc": ["Noraxon MyoMotion-Segments-Shank LT-Acceleration-x (mG)", 
                "Noraxon MyoMotion-Segments-Shank LT-Acceleration-y (mG)", 
                "Noraxon MyoMotion-Segments-Shank LT-Acceleration-z (mG)"],
        "gyro": ["Noraxon MyoMotion-Segments-Shank LT-Gyroscope-x (deg/s)", 
                 "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-y (deg/s)", 
                 "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-z (deg/s)"]
    },
    "Foot_LT": {
        "acc": ["Noraxon MyoMotion-Segments-Foot LT-Acceleration-x (mG)", 
                "Noraxon MyoMotion-Segments-Foot LT-Acceleration-y (mG)", 
                "Noraxon MyoMotion-Segments-Foot LT-Acceleration-z (mG)"],
        "gyro": ["Noraxon MyoMotion-Segments-Foot LT-Gyroscope-x (deg/s)", 
                 "Noraxon MyoMotion-Segments-Foot LT-Gyroscope-y (deg/s)", 
                 "Noraxon MyoMotion-Segments-Foot LT-Gyroscope-z (deg/s)"]
    },
    "Shank_RT": {
        "acc": ["Noraxon MyoMotion-Segments-Shank RT-Acceleration-x (mG)", 
                "Noraxon MyoMotion-Segments-Shank RT-Acceleration-y (mG)", 
                "Noraxon MyoMotion-Segments-Shank RT-Acceleration-z (mG)"],
        "gyro": ["Noraxon MyoMotion-Segments-Shank RT-Gyroscope-x (deg/s)", 
                 "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-y (deg/s)", 
                 "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-z (deg/s)"]
    },
    "Foot_RT": {
        "acc": ["Noraxon MyoMotion-Segments-Foot RT-Acceleration-x (mG)", 
                "Noraxon MyoMotion-Segments-Foot RT-Acceleration-y (mG)", 
                "Noraxon MyoMotion-Segments-Foot RT-Acceleration-z (mG)"],
        "gyro": ["Noraxon MyoMotion-Segments-Foot RT-Gyroscope-x (deg/s)", 
                 "Noraxon MyoMotion-Segments-Foot RT-Gyroscope-y (deg/s)", 
                 "Noraxon MyoMotion-Segments-Foot RT-Gyroscope-z (deg/s)"]
    }
}

def get_highpass_filter(fs=200, cutoff=0.5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    return b, a

def process_file(filepath, b_filter, a_filter):
    try:
        df = pd.read_csv(filepath)
        processed_streams = {}
        
        for sensor, cols in SENSORS_MAP.items():
            # Verify columns exist
            missing = [c for c in cols["acc"] + cols["gyro"] if c not in df.columns]
            if missing:
                # Log warning but don't crash, just skip this sensor or file
                logging.warning(f"File {os.path.basename(filepath)} missing columns: {missing[0]}...")
                return None

            acc = df[cols["acc"]].values
            gyro = df[cols["gyro"]].values
            
            # Gravity Removal & Scaling
            acc_clean = filtfilt(b_filter, a_filter, acc, axis=0)
            acc_norm = acc_clean / 1000.0
            gyro_norm = gyro / 500.0
            
            fused = np.hstack([acc_norm, gyro_norm]).astype(np.float32)
            processed_streams[sensor] = torch.tensor(fused)
            
        return processed_streams
        
    except Exception as e:
        logging.error(f"Failed to process {filepath}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    search_path = os.path.join(INPUT_FOLDER, "S*", "*target_features.csv")
    files = glob.glob(search_path)
    
    logging.info(f"Scanning {os.path.abspath(search_path)}")
    logging.info(f"Found {len(files)} files. Starting processing...")
    
    if len(files) == 0:
        logging.error("No files found! Check path.")
        return

    b, a = get_highpass_filter(fs=200, cutoff=0.5)
    subject_buffer = {}
    
    # --- TQDM PROGRESS BAR ---
    for fpath in tqdm(files, desc="Processing CSVs", unit="file"):
        filename = os.path.basename(fpath)
        
        match = re.match(r"(S\d+)", filename)
        if not match:
            # Try folder name fallback
            parent = os.path.basename(os.path.dirname(fpath))
            if parent.startswith("S"):
                subject_id = parent
            else:
                logging.warning(f"Skipping {filename}: No Subject ID found.")
                continue
        else:
            subject_id = match.group(1)
        
        data_dict = process_file(fpath, b, a)
        
        if data_dict:
            if subject_id not in subject_buffer:
                subject_buffer[subject_id] = {k: [] for k in SENSORS_MAP.keys()}
            
            for sensor in SENSORS_MAP.keys():
                subject_buffer[subject_id][sensor].append(data_dict[sensor])

    logging.info("Consolidating and saving .pt files...")
    
    # Another progress bar for saving
    for sub_id, sensors_data in tqdm(subject_buffer.items(), desc="Saving Tensors", unit="subj"):
        try:
            final_dict = {}
            for sensor in SENSORS_MAP.keys():
                if len(sensors_data[sensor]) > 0:
                    final_dict[sensor] = torch.cat(sensors_data[sensor], dim=0)
                else:
                    logging.warning(f"Subject {sub_id} has no data for {sensor}")
            
            if final_dict:
                save_path = os.path.join(OUTPUT_FOLDER, f"{sub_id}.pt")
                torch.save(final_dict, save_path)
        except Exception as e:
            logging.error(f"Error saving {sub_id}: {e}")

    logging.info(f"Processing Complete. Output: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()