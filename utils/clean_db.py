import os
import glob
import pandas as pd
import shutil
import logging  # Standard logging module
from tqdm import tqdm

# --- CORRECT IMPORT ---
# We import 'setup_logger' from your specific loggers.py file
from utils.loggers import setup_logger

# --- CONFIGURATION ---
# Paths relative to 'verf/' root
DATABASE_ROOT = os.path.join("DATA", "fep","er")
QUARANTINE_ROOT = os.path.join("quarantined_files")
LOG_FILE = "data_cleaning.log"

def clean_database():
	# 1. Initialize Logging using your utility
	setup_logger(LOG_FILE)
	
	# Now we can use standard logging calls
	logging.info(f"Scanning database at: {os.path.abspath(DATABASE_ROOT)}")
	
	# Check if database exists
	if not os.path.exists(DATABASE_ROOT):
		logging.error(f"Folder not found: {DATABASE_ROOT}")
		return

	# Create Quarantine
	if not os.path.exists(QUARANTINE_ROOT):
		os.makedirs(QUARANTINE_ROOT)
		logging.info(f"Created quarantine: {os.path.abspath(QUARANTINE_ROOT)}")

	# Recursive search for all target_features.csv files
	search_pattern = os.path.join(DATABASE_ROOT, "**", "*target_features.csv")
	files = glob.glob(search_pattern, recursive=True)
	
	logging.info(f"Found {len(files)} CSV files. Checking for corruption...")
	
	bad_files = []
	
	for fpath in tqdm(files, desc="Validating CSVs"):
		try:
			# Quick load check
			df = pd.read_csv(fpath, low_memory=False)
			
			# 1. Check for NaN (Missing Values)
			if df.isnull().values.any():
				logging.warning(f"CORRUPT (NaNs): {os.path.basename(fpath)}")
				bad_files.append(fpath)
				continue
				
			# 2. Check for Empty/Short files (< 200 samples)
			if len(df) < 200:
				logging.warning(f"CORRUPT (Too Short): {os.path.basename(fpath)}")
				bad_files.append(fpath)
				continue
				
		except Exception as e:
			logging.error(f"READ ERROR: {os.path.basename(fpath)} | {e}")
			bad_files.append(fpath)

	logging.info("-" * 30)
	
	if len(bad_files) == 0:
		logging.info("✅ No corrupted files found. Your database is clean!")
	else:
		logging.info(f"Found {len(bad_files)} corrupted files.")
		
		# Move files
		moved_count = 0
		for f in bad_files:
			filename = os.path.basename(f)
			# Add parent folder prefix to filename to avoid overwriting duplicates
			parent = os.path.basename(os.path.dirname(f))
			dest_name = f"{parent}_{filename}"
			destination = os.path.join(QUARANTINE_ROOT, dest_name)
			
			try:
				shutil.move(f, destination)
				moved_count += 1
			except Exception as e:
				logging.error(f"Error moving {filename}: {e}")
		
		logging.info(f"Cleanup Complete. Moved {moved_count} files to '{QUARANTINE_ROOT}'.")
		logging.info("You can now safely re-run 'python -m utils.prepare_data'.")

if __name__ == "__main__":
	clean_database()