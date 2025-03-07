import os
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

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
    # Process in groups of 3
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
            # Draw a red circle for the keypoint
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            # Draw the index in blue near the circle
            cv2.putText(img, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def examine_samples(images_dir, labels_dir, num_samples=5, fig_size=(10, 8)):
    """
    Randomly select sample images from images_dir, load corresponding label files from labels_dir,
    annotate them with keypoints, and display using Matplotlib.
    
    Args:
        images_dir (str or Path): Directory with image files.
        labels_dir (str or Path): Directory with label files.
        num_samples (int): Number of random samples to display.
        fig_size (tuple): Figure size (width, height) in inches.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        print("No images found.")
        return
    
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_path in samples:
        # Assume label file has same basename with .txt extension
        label_file = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # If label exists, load and annotate
        if label_file.exists():
            keypoints = load_labels(label_file)
            annotated_img = annotate_image(img.copy(), keypoints)
        else:
            annotated_img = img.copy()
            cv2.putText(annotated_img, "No label file", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert image BGR -> RGB for matplotlib
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Create a Matplotlib window with specified figure size
        plt.figure(figsize=fig_size)
        plt.imshow(annotated_img)
        plt.title(f"Sample: {img_path.name}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Adjust these paths as needed:
    images_dir = "../datasets/train_subset_crop/images"  # Folder with processed images
    labels_dir = "../datasets/train_subset_crop/labels"    # Folder with label files
    examine_samples(images_dir, labels_dir, num_samples=5, fig_size=(10, 8))