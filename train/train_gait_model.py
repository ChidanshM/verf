from utils.data_splitter import create_dataloaders

# ... (Model & Loss definitions) ...

if __name__ == "__main__":
	# Path to your processed .pt files
	DATA_DIR = "verf/processed_tensors"
	
	# Automatically split and load
	train_loader, val_loader, test_data_dict = create_dataloaders(DATA_DIR)
	
	# Initialize Model
	model = SiameseFusion().cuda()
	
	# Train
	for epoch in range(20):
		# ... Training Loop using train_loader ...
		
		# ... Validation Loop (Optional: Check Triplet Loss on Val Set) ...
		
	# After Training: Calculate EER using 'test_data_dict'
	# evaluate_eer(model, test_data_dict)