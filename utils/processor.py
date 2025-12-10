import pandas as pd
import os
import logging

# Define the path to the config file (same directory as this script)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CURRENT_DIR, 'target_columns.csv')

def extract_target_columns(input_file, output_file):
	"""
	Filters CSV for specific biometric columns defined in target_columns.csv.
	"""
	try:
		# 1. Load the target columns from the external CSV
		if not os.path.exists(CONFIG_FILE):
			logging.error(f"Configuration file missing: {CONFIG_FILE}")
			return False, 0, 0
			
		# Read the config file (assuming first column contains the names)
		config_df = pd.read_csv(CONFIG_FILE)
		target_columns = config_df.iloc[:, 0].tolist()

		# 2. Get file size for logging
		orig_size = os.path.getsize(input_file)
		
		# 3. Read the Data (Only loading the columns we need)
		# Note: usecols is faster and saves memory compared to reading the whole file
		df = pd.read_csv(input_file, usecols=lambda c: c in target_columns)
		
		# 4. Reorder columns to match the config list exactly
		# This ensures every file in your dataset has the exact same structure
		existing_cols = [c for c in target_columns if c in df.columns]
		
		# Optional: Warn if columns are missing
		if len(existing_cols) < len(target_columns):
			missing = set(target_columns) - set(existing_cols)
			logging.warning(f"File {os.path.basename(input_file)} is missing {len(missing)} columns.")

		df = df[existing_cols]
		
		# 5. Save
		df.to_csv(output_file, index=False)
		return True, orig_size, os.path.getsize(output_file)

	except Exception as e:
		logging.error(f"Error processing {input_file}: {e}")
		return False, 0, 0