import yaml
import os
from pathlib import Path
from ultralytics.utils.downloads import download
import time
import requests
from tqdm import tqdm
import shutil
import json

def download_with_progress(url, save_path, chunk_size=8192):
    """Download a file with progress bar and resume capability."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get file size
    r = requests.head(url, allow_redirects=True)
    file_size = int(r.headers.get('content-length', 0))
    
    # Check if file already exists for resumption
    initial_pos = 0
    headers = {}
    mode = 'wb'
    
    if os.path.exists(save_path):
        initial_pos = os.path.getsize(save_path)
        if initial_pos < file_size:
            # Resume download
            headers['Range'] = f'bytes={initial_pos}-'
            mode = 'ab'
            print(f"Resuming download from {initial_pos/(1024*1024):.1f} MB")
        else:
            print(f"File already complete: {save_path}")
            return save_path
    
    # Download with progress bar
    print(f"Downloading {url} to {save_path}")
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0)) + initial_pos
        desc = os.path.basename(save_path)
        
        with open(save_path, mode) as f, tqdm(
            desc=desc,
            initial=initial_pos,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)
    
    return save_path

def download_coco_pose(yaml_path="coco-pose.yaml", download_train=True, download_val=True, download_test=False):
    """
    Download COCO-Pose dataset based on the YAML configuration with progress tracking.
    
    Args:
        yaml_path: Path to the YAML configuration file
        download_train: Whether to download training data (19GB)
        download_val: Whether to download validation data (1GB)
        download_test: Whether to download test data (7GB)
    """
    print(f"Loading configuration from {yaml_path}...")
    # Ensure the YAML file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    # Load YAML with utf-8 encoding to avoid character encoding issues
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Set dataset directory
    dataset_dir = Path(data['path'])  # "../datasets/coco-pose"
    print(f"Dataset will be downloaded to: {dataset_dir.absolute()}")
    
    # Create directories if they don't exist
    os.makedirs(dataset_dir.parent, exist_ok=True)
    os.makedirs(dataset_dir / 'images', exist_ok=True)
    
    # Track download progress
    progress_file = dataset_dir / "download_progress.json"
    progress = {"labels": False, "train": False, "val": False, "test": False}
    
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            try:
                progress = json.load(f)
                print("Resuming previous download session")
            except:
                print("Starting new download session")
    
    # Download labels
    if not progress["labels"]:
        print("\nDownloading labels...")
        url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip'
        save_path = str(dataset_dir.parent / "coco2017labels-pose.zip")
        try:
            download_with_progress(url, save_path)
            progress["labels"] = True
            with open(progress_file, "w") as f:
                json.dump(progress, f)
        except Exception as e:
            print(f"Error downloading labels: {str(e)}")
    else:
        print("Labels already downloaded, skipping.")
    
    # Download selected image datasets
    image_downloads = []
    if download_train and not progress["train"]:
        image_downloads.append(("train2017.zip", 'http://images.cocodataset.org/zips/train2017.zip', "train"))
    if download_val and not progress["val"]:
        image_downloads.append(("val2017.zip", 'http://images.cocodataset.org/zips/val2017.zip', "val"))
    if download_test and not progress["test"]:
        image_downloads.append(("test2017.zip", 'http://images.cocodataset.org/zips/test2017.zip', "test"))
    
    if image_downloads:
        for filename, url, dataset_type in image_downloads:
            print(f"\nDownloading {url}...")
            try:
                save_path = str(dataset_dir / 'images' / filename)
                download_with_progress(url, save_path)
                progress[dataset_type] = True
                with open(progress_file, "w") as f:
                    json.dump(progress, f)
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                print("You can resume the download later.")
    else:
        print("No image datasets selected for download or all already downloaded.")
    
    print("\nDownload process complete!")

if __name__ == "__main__":
    download_coco_pose(
        download_train=True,  # 19GB
        download_val=False,    # 1GB
        download_test=False   # 7GB - set to True if you need test data
    )