import os
import numpy as np
from PIL import Image

def create_dummy_data(hospital_id):
    # Define where the data goes
    base_path = os.path.join("data", f"hospital_{hospital_id}")
    
    # Create folders for NORMAL and PNEUMONIA
    for category in ["NORMAL", "PNEUMONIA"]:
        folder_path = os.path.join(base_path, category)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create 10 dummy images per category
        print(f"Generating images for {folder_path}...")
        for i in range(10):
            # Create a random gray square (fake X-ray)
            img_array = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
            img = Image.fromarray(img_array, 'L')
            img.save(os.path.join(folder_path, f"dummy_xray_{i}.jpg"))

if __name__ == "__main__":
    print("--- Generating Dummy Medical Data ---")
    create_dummy_data(1) # For Hospital 1
    create_dummy_data(2) # For Hospital 2
    print("--- Done! Data is ready. ---")