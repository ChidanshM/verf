import os
import zipfile
import shutil
import logging

def find_zip_files(root_dir):
	"""
	Recursively finds ALL .zip files in root_dir.
	Case-insensitive search (finds s001.zip AND S001.zip).
	"""
	zips = []
	# os.walk automatically traverses all subfolders (like set_01, set_02)
	for root, dirs, files in os.walk(root_dir):
		for f in files:
			# Check for .zip extension (case insensitive)
			if f.lower().endswith('.zip'):
				full_path = os.path.join(root, f)
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