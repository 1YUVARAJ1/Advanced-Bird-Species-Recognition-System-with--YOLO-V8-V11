import os
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm

def preprocess_dataset(raw_dir="data/raw/CUB_200_2011", processed_dir="data/processed/CUB_200_2011_cropped"):
    """
    Crops the CUB-200 images using the provided bounding boxes and 
    sorts them into train/test directories based on train_test_split.txt.
    """
    print("Loading dataset annotations...")
    
    # Load metadata
    images = pd.read_csv(os.path.join(raw_dir, "images.txt"), sep=" ", header=None, names=["image_id", "file_name"])
    bboxes = pd.read_csv(os.path.join(raw_dir, "bounding_boxes.txt"), sep=" ", header=None, names=["image_id", "x", "y", "width", "height"])
    split = pd.read_csv(os.path.join(raw_dir, "train_test_split.txt"), sep=" ", header=None, names=["image_id", "is_training_image"])
    
    # Merge dataframes
    df = images.merge(bboxes, on="image_id").merge(split, on="image_id")
    
    # Create target directories
    for split_type in ['train', 'test']:
        split_dir = os.path.join(processed_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)
        
    print(f"Processing {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping Images"):
        image_path = os.path.join(raw_dir, "images", row['file_name'])
        
        # Determine target split directory
        split_type = 'train' if row['is_training_image'] == 1 else 'test'
        class_folder = os.path.dirname(row['file_name'])
        target_dir = os.path.join(processed_dir, split_type, class_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, os.path.basename(row['file_name']))
        
        if os.path.exists(target_path):
            continue # Skip if already processed
            
        try:
            with Image.open(image_path) as img:
                # Get bounding box coordinates for cropping
                # x, y, width, height (yolo uses center, CUB uses top-left)
                left = row['x']
                top = row['y']
                right = left + row['width']
                bottom = top + row['height']
                
                # Crop and save
                cropped_img = img.crop((left, top, right, bottom))
                
                # Convert to RGB if needed (some grayscale images might exist)
                if cropped_img.mode != 'RGB':
                    cropped_img = cropped_img.convert('RGB')
                    
                cropped_img.save(target_path)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    print(f"\nPreprocessing complete. Cropped images saved to {processed_dir}")

if __name__ == "__main__":
    preprocess_dataset()
