import os
import time
import datetime
import logging
import shutil  # Added for removing the temp root
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

# 1. INPUT: Read-Only Source
SOURCE_ROOT = os.path.join(PROJECT_ROOT, 'DATA', 'YA')

# 2. OUTPUT: Final Data Destination
DEST_ROOT = os.path.join(PROJECT_ROOT, 'database','fep')

# 3. TEMP: Temporary Workspace (New!)
# We will extract files here so we never clutter the 'DATA' folder
TEMP_ROOT = os.path.join(PROJECT_ROOT, 'temp_workspace')

# Log Paths
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(PROJECT_ROOT, f"process_log_{TIMESTAMP}.txt")
CSV_FILE = os.path.join(PROJECT_ROOT, f"size_log_{TIMESTAMP}.csv")

def run_feature_selection_pipeline():
	setup_logger(LOG_FILE)
	init_csv_log(CSV_FILE)
	
	print(f"--- Feature Selection Pipeline ---")
	print(f"Input (Read):   {SOURCE_ROOT}")
	print(f"Work Area:      {TEMP_ROOT}")  # Files appear here momentarily
	print(f"Output (Write): {DEST_ROOT}")  # Final files land here

	if not os.path.exists(SOURCE_ROOT):
		print(f"Error: Source directory {SOURCE_ROOT} not found.")
		return

	# Ensure output and temp dirs exist
	if not os.path.exists(DEST_ROOT):
		os.makedirs(DEST_ROOT)
	if not os.path.exists(TEMP_ROOT):
		os.makedirs(TEMP_ROOT)

	# 1. Find Zips
	zip_files = find_zip_files(SOURCE_ROOT)
	
	if not zip_files:
		print("No zip files found.")
		return

	# 2. Outer Loop
	with tqdm(total=len(zip_files), desc="Overall Processing", unit="subj") as pbar_outer:
		
		for zip_path in zip_files:
			subject_id = os.path.basename(zip_path).replace('.zip', '')
			
			# --- PATH LOGIC UPDATE ---
			# Extract to: verf/temp_workspace/s041
			# NOT: verf/DATA/YA/set_01/s041
			temp_extract_path = os.path.join(TEMP_ROOT, subject_id)
			
			# Final Dest: verf/data/s041
			final_dest_path = os.path.join(DEST_ROOT, subject_id)
			
			if not os.path.exists(final_dest_path):
				os.makedirs(final_dest_path)

			try:
				# A. Unzip (into Temp)
				if extract_zip(zip_path, temp_extract_path):
					
					# B. Find Valid CSVs
					valid_csvs = []
					for root, _, files in os.walk(temp_extract_path):
						for file in files:
							# Strict Filter: Only process 'S' files
							if file.lower().endswith('.csv') and file.upper().startswith('S'):
								valid_csvs.append(os.path.join(root, file))
					
					# C. Process Files
					if valid_csvs:
						for input_csv in tqdm(valid_csvs, desc=f"Processing {subject_id}", unit="file", leave=False):
							
							file_name = os.path.basename(input_csv)
							output_csv_name = file_name.replace('.csv', '-raw_targets.csv')
							output_csv = os.path.join(final_dest_path, output_csv_name)

							success, o_size, n_size = extract_target_columns(input_csv, output_csv)
							
							if success:
								log_metric(CSV_FILE, file_name, o_size, output_csv_name, n_size)
					
					# D. Cleanup Subject Temp Folder
					cleanup_folder(temp_extract_path)
					
					logging.info(f"Subject {subject_id} completed.")
			
			except KeyboardInterrupt:
				print("\nAborted.")
				return
			except Exception as e:
				logging.error(f"Error on {subject_id}: {e}")
			
			pbar_outer.update(1)

	# Final cleanup of the main temp folder
	try:
		if os.path.exists(TEMP_ROOT):
			shutil.rmtree(TEMP_ROOT)
	except:
		pass

	print(f"\nDone. Processed data saved to: {DEST_ROOT}")

if __name__ == "__main__":
	run_feature_selection_pipeline()