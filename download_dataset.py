import kaggle
import os
import shutil

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the dataset
print("Downloading vehicle dataset...")
kaggle.api.dataset_download_files('nadinpethiyagoda/vehicle-dataset-for-yolo', path='data', unzip=True)

# Ensure the correct path structure
if os.path.exists('data/vehicle-dataset-for-yolo'):
    # If the dataset was extracted to a different folder, move files to the expected location
    if not os.path.exists('data/vehicle_dataset'):
        os.makedirs('data/vehicle_dataset', exist_ok=True)
    
    # Move contents if needed
    for item in os.listdir('data/vehicle-dataset-for-yolo'):
        src = os.path.join('data/vehicle-dataset-for-yolo', item)
        dst = os.path.join('data/vehicle_dataset', item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

print("Dataset downloaded and extracted to 'data/vehicle_dataset' directory.")
