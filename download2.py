import requests
import os

def download_file(url, save_path):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Get total file size
    file_size = int(response.headers.get('content-length', 0))
    
    # Open the local file to save the download
    with open(save_path, 'wb') as f:
        # Download and write the file in chunks
        chunk_size = 8192
        downloaded = 0
        
        print(f"Downloading {save_path}...")
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Print progress
                progress = (downloaded / file_size) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
    
    print("\nDownload completed!")

# URL of the COCO annotations file
url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# Save path
save_path = "annotations_trainval2017.zip"

# Create the download
download_file(url, save_path)