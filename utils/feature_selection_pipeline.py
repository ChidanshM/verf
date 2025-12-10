import os
import time
import datetime
import logging
import zipfile
from tqdm import tqdm

# Import from sibling modules
try:
	from processor import extract_target_columns
	from filesystem import find_zip_files
	from loggers import setup_logger, init_csv_log, log_metric
except ImportError:
	from .processor import extract_target_columns
	from .filesystem import find_zip_files
	from .loggers import setup_logger, init_csv_log, log_metric

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

# 1. INPUT: Read-Only Source (Looking in verf/DATA/ma)
SOURCE_ROOT = os.path.join(PROJECT_ROOT, 'DATA', 'ma')

# 2. OUTPUT: Final Data Destination
DEST_ROOT = os.path.join(PROJECT_ROOT, 'database', 'fep')

# Log Paths
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(PROJECT_ROOT, f"process_log_{TIMESTAMP}.txt")
CSV_FILE = os.path.join(PROJECT_ROOT, f"size_log_{TIMESTAMP}.csv")

def run_feature_selection_pipeline():
	setup_logger(LOG_FILE)
	init_csv_log(CSV_FILE)
	
	print(f"--- Feature Selection Pipeline ---")
	print(f"Input (Read):   {SOURCE_ROOT}")
	print(f"Output (Write): {DEST_ROOT}")  
	print(f"Mode:           In-Memory Processing (No Temp Folder)")

	if not os.path.exists(SOURCE_ROOT):
		print(f"Error: Source directory {SOURCE_ROOT} not found.")
		return

	# Ensure output dir exists
	if not os.path.exists(DEST_ROOT):
		os.makedirs(DEST_ROOT)

	# 1. Find Zips
	zip_files = find_zip_files(SOURCE_ROOT)
	
	if not zip_files:
		print(f"No zip files found in {SOURCE_ROOT}")
		return

	# 2. Processing Loop
	with tqdm(total=len(zip_files), desc="Overall Processing", unit="subj") as pbar_outer:
		
		for zip_path in zip_files:
			subject_id = os.path.basename(zip_path).replace('.zip', '')
			final_dest_path = os.path.join(DEST_ROOT, subject_id)
			
			if not os.path.exists(final_dest_path):
				os.makedirs(final_dest_path)

			try:
				# Open Zip (No extraction to disk)
				with zipfile.ZipFile(zip_path, 'r') as z:
					
					# Find valid CSVs inside the zip
					# We look for files ending in .csv and starting with 'S' (case insensitive check)
					file_list = z.infolist()
					valid_files = [f for f in file_list if f.filename.lower().endswith('.csv') and f.filename.upper().startswith('S')]
					
					if valid_files:
						for zip_info in tqdm(valid_files, desc=f"Processing {subject_id}", unit="file", leave=False):
							
							file_name = os.path.basename(zip_info.filename)
							output_csv_name = file_name.replace('.csv', '-target_features.csv')
							output_csv_path = os.path.join(final_dest_path, output_csv_name)

							# OPEN FILE IN MEMORY
							with z.open(zip_info) as file_obj:
								# Pass the file object + original size to the processor
								success, o_size, n_size = extract_target_columns(
									input_source=file_obj, 
									output_file=output_csv_path,
									input_size=zip_info.file_size
								)
							
							if success:
								log_metric(CSV_FILE, file_name, o_size, output_csv_name, n_size)
					
					logging.info(f"Subject {subject_id} completed.")
			
			except KeyboardInterrupt:
				print("\nAborted.")
				return
			except Exception as e:
				logging.error(f"Error on {subject_id}: {e}")
			
			pbar_outer.update(1)

	print(f"\nDone. Processed data saved to: {DEST_ROOT}")

if __name__ == "__main__":
	run_feature_selection_pipeline()