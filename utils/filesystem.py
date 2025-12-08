import os
import zipfile
import shutil
import logging

def find_zip_files(root_dir):
	"""
	STRICT SEARCH:
	1. Looks in 'root_dir' (DATA/YA)
	2. Selects ONLY folders named 'set_...'
	3. Inside those, selects ONLY files named 's...zip' or 'S...zip'
	"""
	zips = []
	print(f"DEBUG: Searching for zips in {root_dir}...")
	
	if not os.path.exists(root_dir):
		return []

	# 1. Get all items in DATA/YA
	level_1_items = os.listdir(root_dir)
	
	for folder_name in level_1_items:
		folder_path = os.path.join(root_dir, folder_name)
		
		# RULE 1: Must be a Directory AND contain "set_"
		if os.path.isdir(folder_path) and "set_" in folder_name.lower():
			
			# 2. Look inside the set_## folder
			set_folder_items = os.listdir(folder_path)
			
			for file_name in set_folder_items:
				# RULE 2: Must be a Zip AND start with 's'
				if file_name.lower().startswith('s') and file_name.lower().endswith('.zip'):
					full_path = os.path.join(folder_path, file_name)
					zips.append(full_path)
					
	return zips

def extract_zip(zip_path, extract_to):
	"""Unzips file to destination."""
	try:
		if not os.path.exists(extract_to):
			os.makedirs(extract_to)
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(extract_to)
		return True
	except Exception as e:
		logging.error(f"Unzip failed for {zip_path}: {e}")
		return False

def cleanup_folder(folder_path):
	"""Safely deletes a directory."""
	try:
		if os.path.exists(folder_path):
			shutil.rmtree(folder_path)
			logging.info(f"Cleaned up {folder_path}")
	except Exception as e:
		logging.error(f"Cleanup failed for {folder_path}: {e}")