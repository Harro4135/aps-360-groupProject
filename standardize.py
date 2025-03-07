import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random

def standardize_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Resize all .jpg images in input_dir to target size and save to output_dir.
    
    Args:
      input_dir (str or Path): Directory containing the images.
      output_dir (str or Path): Directory where resized images will be saved.
      target_size (tuple): Desired output size (width, height). Default is 224x224.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = list(input_dir.glob("*.jpg"))
    if not image_files:
        print("No images found in the input directory.")
        return
    
    count = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        # Resize image to target_size (cv2.resize expects (width, height))
        resized = cv2.resize(img, target_size)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), resized)
        count += 1
    
    print(f"Resized and saved {count} images to {output_dir}")

def load_labels(label_path):
    """
    Read a label file and parse it.
    Expected format: "0 norm_x norm_y v norm_x norm_y v ..."
    Returns: list of tuples [(x, y, v), ...]
    """
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    # Ignore the first token (class id)
    kp_values = parts[1:]
    keypoints = []
    for i in range(0, len(kp_values), 3):
        try:
            x = float(kp_values[i])
            y = float(kp_values[i+1])
            v = int(kp_values[i+2])
            keypoints.append((x, y, v))
        except Exception as e:
            print(f"Error parsing keypoints: {e}")
    return keypoints

def annotate_image(img, keypoints):
    """
    Draw keypoints on the image.
    The keypoints are in normalized format and need to be scaled to image dimensions.
    Only draw keypoints that are labeled (v > 0).
    """
    H, W = img.shape[:2]
    for idx, (nx, ny, v) in enumerate(keypoints):
        if v > 0:
            x = int(nx * W)
            y = int(ny * H)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(img, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def compare_images(original_dir, standardized_dir, num_samples=5, figsize=(12, 6),
                   original_labels_dir=None, standardized_labels_dir=None):
    """
    Display pairs of original and standardized images side by side with optional label annotation.
    
    Args:
        original_dir (str or Path): Directory containing the original images.
        standardized_dir (str or Path): Directory containing the standardized images.
        num_samples (int): Number of image pairs to display.
        figsize (tuple): Figure size for the Matplotlib window.
        original_labels_dir (str or Path, optional): Directory containing label files for the original images.
        standardized_labels_dir (str or Path, optional): Directory containing label files for the standardized images.
    """
    original_dir = Path(original_dir)
    standardized_dir = Path(standardized_dir)
    
    # Convert label directories to Path objects if provided.
    if original_labels_dir is not None:
        original_labels_dir = Path(original_labels_dir)
    if standardized_labels_dir is not None:
        standardized_labels_dir = Path(standardized_labels_dir)
    
    # Gather only those images that have a matching standardized version.
    original_files = list(original_dir.glob("*.jpg"))
    paired_files = []
    for img_path in original_files:
        std_path = standardized_dir / img_path.name
        if std_path.exists():
            paired_files.append((img_path, std_path))
    
    if not paired_files:
        print("No matching image pairs were found.")
        return
    
    samples = random.sample(paired_files, min(num_samples, len(paired_files)))
    
    for orig_path, std_path in samples:
        # Load images.
        orig_img = cv2.imread(str(orig_path))
        std_img = cv2.imread(str(std_path))
        if orig_img is None or std_img is None:
            continue
        
        # Optionally annotate original image if label file exists.
        if original_labels_dir is not None:
            orig_lab_file = original_labels_dir / (orig_path.stem + ".txt")
            if orig_lab_file.exists():
                orig_labels = load_labels(orig_lab_file)
                orig_img = annotate_image(orig_img.copy(), orig_labels)
        
        # Optionally annotate standardized image if label file exists.
        if standardized_labels_dir is not None:
            std_lab_file = standardized_labels_dir / (std_path.stem + ".txt")
            if std_lab_file.exists():
                std_labels = load_labels(std_lab_file)
                std_img = annotate_image(std_img.copy(), std_labels)
        
        # Convert images from BGR to RGB for Matplotlib.
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        std_img = cv2.cvtColor(std_img, cv2.COLOR_BGR2RGB)
        
        # Create a side-by-side plot with a super title.
        plt.figure(figsize=figsize)
        plt.suptitle("Comparison of Original and Standardized Images", fontsize=16)
        
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title(f"Original: {orig_path.name}")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(std_img)
        plt.title(f"Standardized: {std_path.name}")
        plt.axis("off")
        
        plt.show()

if __name__ == "__main__":
    original_dir = "../datasets/train_subset_single/images"  # Directory with original images.
    standardized_dir = "../datasets/train_subset_single/standardized_images"  # Directory with standardized images.
    # Optionally, provide label directories; if not, pass None.
    original_labels_dir = "../datasets/train_subset_single/labels"
    standardized_labels_dir = "../datasets/train_subset_single/labels"  

    standardize_images(original_dir, standardized_dir, target_size=(220, 220))
    
    compare_images(original_dir, standardized_dir, num_samples=5, figsize=(12, 6),
                   original_labels_dir=original_labels_dir,
                   standardized_labels_dir=standardized_labels_dir)