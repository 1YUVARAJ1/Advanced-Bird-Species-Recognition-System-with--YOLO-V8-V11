import os
import tarfile
import urllib.request
import sys

def download_progress_hook(block_num, block_size, total_size):
    if total_size > 0:
        percent = min((block_num * block_size * 100) / total_size, 100)
        sys.stdout.write(f"\rDownloading... {percent:.1f}%")
        sys.stdout.flush()

def setup_dataset():
    dataset_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz" # Alternative official mirror if main is down
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    tar_path = os.path.join(raw_dir, "CUB_200_2011.tgz")
    extracted_dir = os.path.join(raw_dir, "CUB_200_2011")
    
    if os.path.exists(extracted_dir):
        print("Dataset already extracted in", extracted_dir)
        return
        
    if not os.path.exists(tar_path):
        print("Downloading CUB-200-2011 dataset (~1.1GB)...")
        try:
            urllib.request.urlretrieve(dataset_url, tar_path, reporthook=download_progress_hook)
            print("\nDownload complete.")
        except Exception as e:
            print("\nDownload failed. Please download it manually from Kaggle or Caltech.")
            print("Place the extracted CUB_200_2011 folder inside AdvancedBirdRecognition/data/raw/")
            print(f"Error: {e}")
            return
            
    print("Extracting dataset...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=raw_dir)
        print("Extraction complete.")
    except Exception as e:
        print("Failed to extract:", e)

if __name__ == "__main__":
    setup_dataset()
