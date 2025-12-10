import pandas as pd
import os
import logging
import io

# Define the path to the config file (same directory as this script)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CURRENT_DIR, 'target_columns.csv')

def extract_target_columns(input_source, output_file, input_size=0):
	"""
	Filters CSV for specific biometric columns defined in target_columns.csv.
	
	Args:
		input_source: Can be a file path (str) OR a file-like object (from zip)
		output_file: Path to save the processed CSV
		input_size: Size of original file (passed in manually if reading from zip)
	"""
	try:
		# 1. Load the target columns from the external CSV
		if not os.path.exists(CONFIG_FILE):
			logging.error(f"Configuration file missing: {CONFIG_FILE}")
			return False, 0, 0
			
		# Read the config file (assuming first column contains the names)
		config_df = pd.read_csv(CONFIG_FILE)
		target_columns = config_df.iloc[:, 0].tolist()

		# 2. Handle Input Source (Path vs File Object)
		if isinstance(input_source, str):
			# It's a file path
			if os.path.exists(input_source):
				input_size = os.path.getsize(input_source)
			# pandas can read the path directly
			df = pd.read_csv(input_source, usecols=lambda c: c in target_columns)
		else:
			# It's a file object (from zip)
			# pandas can read the file object directly
			df = pd.read_csv(input_source, usecols=lambda c: c in target_columns)
		
		# 3. Reorder columns to match the config list exactly
		existing_cols = [c for c in target_columns if c in df.columns]
		
		# Optional: Warn if columns are missing
		if len(existing_cols) < len(target_columns):
			missing = set(target_columns) - set(existing_cols)
			# Only warn if we can get a filename, otherwise generic warning
			name = getattr(input_source, 'name', 'unknown_file')
			logging.warning(f"File {name} is missing {len(missing)} columns.")

		df = df[existing_cols]
		
		# 4. Save
		df.to_csv(output_file, index=False)
		
		# Calculate new size
		output_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
		
		return True, input_size, output_size

	except Exception as e:
		name = getattr(input_source, 'name', 'unknown_file')
		logging.error(f"Error processing {name}: {e}")
		return False, 0, 0