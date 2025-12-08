import logging
import csv
import sys

def setup_logger(log_file):
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
	)

def init_csv_log(csv_file):
	with open(csv_file, 'w', newline='') as f:
		csv.writer(f).writerow(['Original', 'Orig_Bytes', 'Processed', 'Proc_Bytes'])

def log_metric(csv_file, *args):
	with open(csv_file, 'a', newline='') as f:
		csv.writer(f).writerow(args)