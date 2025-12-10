import os
import zipfile
import shutil
import logging

def find_zip_files(root_dir):
	"""
	STRICT SEARCH:
	1. Looks in 'root_dir' (DATA/YA)
	3. Inside those, selects ONLY files named 's...zip' or 'S...zip'
	"""
	zips = []
	print(f"DEBUG: Searching for zips in {root_dir}...")
	
	if not os.path.exists(root_dir):
		print(f"DEBUG: Directory {root_dir} does not exist.")
		return

	# 1. Get all items in DATA/YA
	level_1_items = os.listdir(root_dir)
	try:
		items = os.listdir(root_dir)
	except Exception as e:
		logging.error(f"Failed to list directory {root_dir}: {e}")
		return []
    
	for file_name in items:
		# Check if it is a zip file
		if file_name.lower().endswith('.zip'):
			full_path = os.path.join(root_dir, file_name)
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