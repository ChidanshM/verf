import pandas as pd
import os
import logging

def extract_target_columns(input_file, output_file):
	"""Filters CSV for specific biometric columns."""
	target_columns = [
		'time',
		'Pelvis Accel Sensor X (mG)', 'Pelvis Accel Sensor Y (mG)', 'Pelvis Accel Sensor Z (mG)',
		'Upper spine Accel Sensor X (mG)', 'Upper spine Accel Sensor Y (mG)', 'Upper spine Accel Sensor Z (mG)',
		'Shank Accel Sensor X LT (mG)', 'Shank Accel Sensor Y LT (mG)', 'Shank Accel Sensor Z LT (mG)',
		'Foot Accel Sensor X LT (mG)', 'Foot Accel Sensor Y LT (mG)', 'Foot Accel Sensor Z LT (mG)',
		'Shank Accel Sensor X RT (mG)', 'Shank Accel Sensor Y RT (mG)', 'Shank Accel Sensor Z RT (mG)',
		'Foot Accel Sensor X RT (mG)', 'Foot Accel Sensor Y RT (mG)', 'Foot Accel Sensor Z RT (mG)'
	]
	try:
		orig_size = os.path.getsize(input_file)
		df = pd.read_csv(input_file, usecols=lambda c: c in target_columns)
		
		# Reorder to match target list
		existing_cols = [c for c in target_columns if c in df.columns]
		df = df[existing_cols]
		
		df.to_csv(output_file, index=False)
		return True, orig_size, os.path.getsize(output_file)
	except Exception as e:
		logging.error(f"Error in {input_file}: {e}")
		return False, 0, 0