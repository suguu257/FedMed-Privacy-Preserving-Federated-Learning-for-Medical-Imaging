import os
import shutil
import random

# Setup paths
original_dataset_dir = './dataset/train'
base_dir = './clients'

# We will create two imaginary hospitals
hospitals = ['hospital_1', 'hospital_2']
categories = ['NORMAL', 'PNEUMONIA']

# Create the folders if they don't exist
for hospital in hospitals:
    for category in categories:
        dir_path = os.path.join(base_dir, hospital, category)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created folder: {dir_path}")

# Function to split data
def split_data():
    print("Splitting data between hospitals... please wait.")
    
    for category in categories:
        # Get list of all images in the original folder
        src_path = os.path.join(original_dataset_dir, category)
        filenames = os.listdir(src_path)
        
        # Shuffle them so it's random
        random.shuffle(filenames)
        
        # Split: First half goes to Hospital 1, second half to Hospital 2
        split_point = len(filenames) // 2
        
        hospital_1_files = filenames[:split_point]
        hospital_2_files = filenames[split_point:]
        
        # Copy files to Hospital 1
        for fname in hospital_1_files:
            src = os.path.join(src_path, fname)
            dst = os.path.join(base_dir, 'hospital_1', category, fname)
            shutil.copyfile(src, dst)
            
        # Copy files to Hospital 2
        for fname in hospital_2_files:
            src = os.path.join(src_path, fname)
            dst = os.path.join(base_dir, 'hospital_2', category, fname)
            shutil.copyfile(src, dst)
            
        print(f"Finished splitting {category} images.")

if __name__ == "__main__":
    split_data()
    print("Done! You now have two separate hospitals with their own private data.")