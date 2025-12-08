import pandas as pd
import os

def extract_target_columns(input_file, output_file):
    """
    Reads a full NONAN GaitPrint file and saves a new CSV containing
    ONLY the raw acceleration columns for the selected biometric sensors.
    """
    
    # 1. Define the EXACT column names (based on column_names.csv)
    # These are the raw X, Y, Z acceleration columns for your 6 sensors.
    target_columns = [
        'time',  # Essential for sequence
        
        # --- Core Stability (Torso) ---
        'Pelvis Accel Sensor X (mG)',
        'Pelvis Accel Sensor Y (mG)',
        'Pelvis Accel Sensor Z (mG)',
        
        'Upper spine Accel Sensor X (mG)', # Maps to Upper Thoracic
        'Upper spine Accel Sensor Y (mG)',
        'Upper spine Accel Sensor Z (mG)',
        
        # --- Lower Body Chain (Left) ---
        'Shank Accel Sensor X LT (mG)',
        'Shank Accel Sensor Y LT (mG)',
        'Shank Accel Sensor Z LT (mG)',
        
        'Foot Accel Sensor X LT (mG)',
        'Foot Accel Sensor Y LT (mG)',
        'Foot Accel Sensor Z LT (mG)',
        
        # --- Lower Body Chain (Right) ---
        'Shank Accel Sensor X RT (mG)',
        'Shank Accel Sensor Y RT (mG)',
        'Shank Accel Sensor Z RT (mG)',
        
        'Foot Accel Sensor X RT (mG)',
        'Foot Accel Sensor Y RT (mG)',
        'Foot Accel Sensor Z RT (mG)'
    ]

    print(f"Reading file: {input_file}...")

    try:
        # 2. Load ONLY the target columns
        # 'usecols' acts as a filter, ignoring the other ~300 columns immediately
        df = pd.read_csv(input_file, usecols=target_columns)
        
        # 3. Reorder columns (Optional, for tidiness)
        # Ensures the output CSV follows the exact order defined above
        df = df[target_columns]

        # 4. Save to New CSV
        df.to_csv(output_file, index=False)
        
        print(f"Success! Created {output_file}")
        print(f"Output Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
        print("\n--- Preview of Output Data ---")
        print(df.head())

    except ValueError as e:
        print(f"Error: A column was not found in the input file.\nDetails: {e}")
        print("Tip: Check if the input file has the correct headers (Row 1).")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")

# --- USAGE ---
# 1. Update this path to your downloaded subject file
input_path = 'S001_G01_D01_B01_T01.csv' 

# 2. Define your output name
output_path = 'S001_G01_D01_B01_T01-Raw_Targets_Only.csv'

# 3. Run
if __name__ == "__main__":
    # create dummy file for demonstration if not present
    if not os.path.exists(input_path):
        print("Note: Input file not found, creating dummy file for testing...")
        dummy_cols = [
            'time', 
            'Pelvis Accel Sensor X (mG)', 'Pelvis Accel Sensor Y (mG)', 'Pelvis Accel Sensor Z (mG)',
            'Upper spine Accel Sensor X (mG)', 'Upper spine Accel Sensor Y (mG)', 'Upper spine Accel Sensor Z (mG)',
            'Shank Accel Sensor X LT (mG)', 'Shank Accel Sensor Y LT (mG)', 'Shank Accel Sensor Z LT (mG)',
            'Foot Accel Sensor X LT (mG)', 'Foot Accel Sensor Y LT (mG)', 'Foot Accel Sensor Z LT (mG)',
            'Shank Accel Sensor X RT (mG)', 'Shank Accel Sensor Y RT (mG)', 'Shank Accel Sensor Z RT (mG)',
            'Foot Accel Sensor X RT (mG)', 'Foot Accel Sensor Y RT (mG)', 'Foot Accel Sensor Z RT (mG)',
            'Extra_Junk_Column_1', 'Extra_Junk_Column_2' # Simulating the extra 300 cols
        ]
        pd.DataFrame(columns=dummy_cols).to_csv(input_path, index=False)
        
    extract_target_columns(input_path, output_path)