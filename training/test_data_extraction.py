import os
import shutil
from glob import glob
import random

def move_test_images(dataset_dir, test_dataset_dir, test_ratio=0.2):
    # Ensure the test dataset directory exists
    os.makedirs(test_dataset_dir, exist_ok=True)

    # Iterate over each directory within the dataset directory
    for dir_path in glob(os.path.join(dataset_dir, '*')):
        if os.path.isdir(dir_path):
            # For each 'REAL' and 'FAKE' folder
            for label in ['REAL', 'FAKE']:
                full_label_path = os.path.join(dir_path, label)
                if os.path.exists(full_label_path) and os.path.isdir(full_label_path):
                    # List all images
                    images = [img for img in os.listdir(full_label_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
                    random.shuffle(images)  # Shuffle to randomize which images are selected

                    # Calculate the number of test images to move
                    num_test_images = int(len(images) * test_ratio)

                    # Create the corresponding directory in the test_dataset directory
                    test_dir_path = os.path.join(test_dataset_dir,label)
                    os.makedirs(test_dir_path, exist_ok=True)

                    # Move the images
                    for img in images[:num_test_images]:
                        src_path = os.path.join(full_label_path, img)
                        dst_path = os.path.join(test_dir_path, img)
                        shutil.move(src_path, dst_path)
                    print(f"Moved {num_test_images} images from {full_label_path} to {test_dir_path}")

dataset_dir = r'./dataset'  # Your dataset directory
test_dataset_dir = r'./test_dataset2'  # The directory where test images will be stored

move_test_images(dataset_dir, test_dataset_dir)