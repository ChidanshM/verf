import os
import time
import datetime
import logging
from tqdm import tqdm

# Import from sibling modules
try:
	from processor import extract_target_columns
	from filesystem import find_zip_files, extract_zip, cleanup_folder
	from loggers import setup_logger, init_csv_log, log_metric
except ImportError:
	from .processor import extract_target_columns
	from .filesystem import find_zip_files, extract_zip, cleanup_folder
	from .loggers import setup_logger, init_csv_log, log_metric

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

SOURCE_ROOT = os.path.join(PROJECT_ROOT, 'DATA', 'YA')
DEST_ROOT = os.path.join(PROJECT_ROOT, 'data')

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(PROJECT_ROOT, f"process_log_{TIMESTAMP}.txt")
CSV_FILE = os.path.join(PROJECT_ROOT, f"size_log_{TIMESTAMP}.csv")

def run_feature_selection_pipeline():
	# Setup File Logging ONLY (No console handler to avoid messing up tqdm)
	logging.basicConfig(
		filename=LOG_FILE,
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
	)
	init_csv_log(CSV_FILE)
	
	print(f"--- Feature Selection Pipeline ---")
	print(f"Logs: {LOG_FILE}")

	if not os.path.exists(SOURCE_ROOT):
		print(f"Error: Source {SOURCE_ROOT} not found.")
		return

	# 1. Get Work List
	zip_files = find_zip_files(SOURCE_ROOT)
	
	if not zip_files:
		print("No zip files found.")
		return

	# 2. Outer Loop: Overall Processing
	# This bar stays visible until the end
	with tqdm(total=len(zip_files), desc="Overall Processing", unit="subj") as pbar_outer:
		
		for zip_path in zip_files:
			subject_id = os.path.basename(zip_path).replace('.zip', '')
			
			# Paths
			temp_extract_path = os.path.join(os.path.dirname(zip_path), subject_id)
			final_dest_path = os.path.join(DEST_ROOT, subject_id)
			
			if not os.path.exists(final_dest_path):
				os.makedirs(final_dest_path)

			try:
				# A. Unzip
				if extract_zip(zip_path, temp_extract_path):
					
					# B. Find Valid CSVs
					valid_csvs = []
					for root, _, files in os.walk(temp_extract_path):
						for file in files:
							# STRICT FILTER: Only files starting with 'S' (Subject Data)
							# This ignores 'Gaitprint_Noraxon...' and other junk inside the zip
							if file.lower().endswith('.csv') and file.upper().startswith('S'):
								valid_csvs.append(os.path.join(root, file))
					
					# Inner Loop: Single Subject Processing
					# This bar appears for 1.5 mins then disappears (leave=False)
					if valid_csvs:
						for input_csv in tqdm(valid_csvs, desc=f"Processing {subject_id}", unit="file", leave=False):
							
							file_name = os.path.basename(input_csv)
							output_csv_name = file_name.replace('.csv', '-raw_targets.csv')
							output_csv = os.path.join(final_dest_path, output_csv_name)

							success, o_size, n_size = extract_target_columns(input_csv, output_csv)
							
							if success:
								# Log to file ONLY, not console
								log_metric(CSV_FILE, file_name, o_size, output_csv_name, n_size)
					
					# C. Cleanup
					cleanup_folder(temp_extract_path)
					
					logging.info(f"Subject {subject_id} completed.")
			
			except KeyboardInterrupt:
				print("\nAborted.")
				return
			except Exception as e:
				logging.error(f"Error on {subject_id}: {e}")
			
			# Update Outer Bar
			pbar_outer.update(1)

	print(f"\nDone. Processed data saved to: {DEST_ROOT}")

if __name__ == "__main__":
	run_feature_selection_pipeline()