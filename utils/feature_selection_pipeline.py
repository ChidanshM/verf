# ... (Imports remain the same) ...
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

# Log Paths
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(PROJECT_ROOT, f"process_log_{TIMESTAMP}.txt")
CSV_FILE = os.path.join(PROJECT_ROOT, f"size_log_{TIMESTAMP}.csv")

def run_feature_selection_pipeline():
	setup_logger(LOG_FILE)
	init_csv_log(CSV_FILE)
	
	# --- DIAGNOSTIC PRINT ---
	abs_source = os.path.abspath(SOURCE_ROOT)
	print(f"--- Feature Selection Pipeline Started ---")
	print(f"Looking for data in: {abs_source}")  # <--- CHECK THIS PATH IN OUTPUT
	print(f"Writing output to:   {os.path.abspath(DEST_ROOT)}")
	
	if not os.path.exists(SOURCE_ROOT):
		logging.error(f"Directory not found: {abs_source}")
		print(f"\n[ERROR] Python cannot see the folder: {abs_source}")
		print("Please check that your 'DATA' folder is spelled exactly right (case sensitive).")
		return

	# 2. Get Work List
	zip_files = find_zip_files(SOURCE_ROOT)
	print(f"Found {len(zip_files)} zip files.")
	logging.info(f"Found {len(zip_files)} subject archives to process.")

	# 3. Processing Loop
	if len(zip_files) == 0:
		print("\n[WARNING] No zip files found!")
		print("Check: Are the files inside subfolders? Are they definitely .zip files?")
		return

	for zip_path in tqdm(zip_files, desc="Processing Subjects", unit="subj"):
		# ... (Rest of the loop logic remains identical to previous version) ...
		# ... (Include the Try/Except block, Inner loop, Cleanup, etc.) ...
		
		subject_id = os.path.basename(zip_path).replace('.zip', '')
		
		# Define paths
		temp_extract_path = os.path.join(os.path.dirname(zip_path), subject_id)
		final_dest_path = os.path.join(DEST_ROOT, subject_id)
		
		if not os.path.exists(final_dest_path):
			os.makedirs(final_dest_path)

		start_time = time.time()

		try:
			# A. Unzip
			if extract_zip(zip_path, temp_extract_path):
				
				# B. Gather CSVs first (so we can make a progress bar)
				csv_files_to_process = []
				for root, _, files in os.walk(temp_extract_path):
					for file in files:
						if file.lower().endswith('.csv'):
							csv_files_to_process.append(os.path.join(root, file))
				
				# Inner Processing Loop (Inner Progress Bar)
				# leave=False makes this bar disappear when the subject is done
				for input_csv in tqdm(csv_files_to_process, desc=f"  Trials ({subject_id})", unit="file", leave=False):
					
					file_name = os.path.basename(input_csv)

					# Naming convention: s001_...-raw_targets.csv
					output_csv_name = file_name.replace('.csv', '-raw_targets.csv')
					output_csv = os.path.join(final_dest_path, output_csv_name)

					# CALL PROCESSOR
					success, o_size, n_size = extract_target_columns(input_csv, output_csv)
					
					if success:
						logging.info(f"Selected Features: {file_name}")
						log_metric(CSV_FILE, file_name, o_size, output_csv_name, n_size)

				# C. Cleanup
				cleanup_folder(temp_extract_path)
				
				duration = time.time() - start_time
				logging.info(f"Subject {subject_id} completed in {duration:.2f}s")
		
		except KeyboardInterrupt:
			print("\nProcess interrupted by user.")
			logging.warning("Process interrupted by user.")
			break
		except Exception as e:
			logging.error(f"Unexpected error processing {subject_id}: {e}")

	logging.info("--- Pipeline Finished ---")
	print(f"\nDone. Logs saved to: {PROJECT_ROOT}")

if __name__ == "__main__":
	run_feature_selection_pipeline()