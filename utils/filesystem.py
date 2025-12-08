#verf/utils/filesystem.py
import os
import zipfile
import shutil
import logging

def find_zip_files(root_dir):
	zips = []
	for root, dirs, files in os.walk(root_dir):
		if "set_" in os.path.basename(root):
			for f in files:
				if f.endswith('.zip') and f.startswith('s'):
					zips.append(os.path.join(root, f))
	return zips

def extract_zip(zip_path, extract_to):
	try:
		with zipfile.ZipFile(zip_path, 'r') as z:
			z.extractall(extract_to)
		return True
	except Exception as e:
		logging.error(f"Unzip failed: {e}")
		return False

def cleanup_folder(path):
	try:
		shutil.rmtree(path)
	except Exception as e:
		logging.error(f"Cleanup failed: {e}")