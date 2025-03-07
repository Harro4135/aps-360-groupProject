import os
import json
import shutil
import cv2
from pathlib import Path

def normalize_keypoints(keypoints, width, height):
    """
    Given a flat list of keypoints [x1, y1, v1, ...],
    normalize the x and y coordinates relative to width and height.
    Returns a flat list of normalized keypoints.
    """
    num_kps = len(keypoints) // 3
    norm_kps = []
    for i in range(num_kps):
        x = keypoints[3*i]
        y = keypoints[3*i+1]
        v = keypoints[3*i+2]
        norm_x = x / width
        norm_y = y / height
        norm_kps.extend([norm_x, norm_y, v])
    return norm_kps

def write_label_file(label_path, keypoints):
    """
    Write keypoints (YOLO-like format) into a label file.
    The label follows the format:
        0 norm_x norm_y v norm_x norm_y v ...
    """
    label_line = "0 " + " ".join(
        f"{val:.6f}" if isinstance(val, float) else str(val)
        for val in keypoints
    )
    with open(label_path, 'w') as f:
        f.write(label_line + "\n")

def filter_keypoints(keypoints):
    """
    If the keypoints list is for 17 keypoints (i.e. length == 51),
    remove keypoints 1,2,3,4 (i.e. skip keypoints with indices 1,2,3,4)
    but keep keypoint 0, so that only the remaining 13 keypoints are kept.
    Then count the number of visible keypoints (visibility flag > 0).
    Return the filtered keypoints if at least two are visible.
    If not, return None.
    If the list length is not 51, leave it unchanged (but still require at least 2 visible).
    """
    num_kps = len(keypoints) // 3
    if num_kps == 17:
        # Keep keypoint 0 (indices 0:3) and keypoints 5..16 (indices 15:51)
        filtered = keypoints[0:3] + keypoints[15:]
    else:
        filtered = keypoints

    visible = sum(1 for i in range(len(filtered)//3) if filtered[3*i+2] > 0)
    if visible <= 2:
        return None
    return filtered

def process_pipeline(json_path, images_dir, output_dir):
    """
    Process images from a COCO person_keypoints JSON *only* for images that contain one person.
    
    For each single-person image (non-crowd):
      - Modify the keypoints: if there are 17 keypoints, remove keypoints 1-4,
        then check that at least two keypoints are visible.
      - Copy the full image and generate a label file using normalized keypoints (relative to the full image).
      - Skip the image entirely if, after filtering, there are insufficient visible keypoints.
    
    Args:
      json_path (str): Path to the COCO annotation JSON file.
      images_dir (str or Path): Directory containing the source images.
      output_dir (str or Path): Directory to save the processed subset.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Build mapping from image_id to image metadata.
    image_map = {img['id']: img for img in data.get('images', [])}
    
    # Group annotations by image_id for category_id == 1.
    img_to_anns = {}
    for ann in data.get('annotations', []):
        if ann.get('category_id') == 1:
            img_to_anns.setdefault(ann['image_id'], []).append(ann)
    
    single_count = 0

    # Only process images with exactly one person.
    for image_id, anns in img_to_anns.items():
        if len(anns) != 1:
            continue  # Skip images with multiple persons
        
        ann = anns[0]
        if ann.get('iscrowd', 0) == 1:
            continue  # Skip if "iscrowd" is set.
        
        img_info = image_map.get(image_id)
        if not img_info:
            continue
        
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error reading image: {img_path}")
            continue
        
        H, W = image.shape[:2]
        keypoints = ann.get('keypoints', [])
        if len(keypoints) % 3 != 0:
            continue  # Skip if keypoints not valid.
        
        # Filter the keypoints to remove keypoints 1-4 if applicable and check visibility.
        filtered_kps = filter_keypoints(keypoints)
        if filtered_kps is None:
            continue  # Skip image if too few visible keypoints
        
        # Normalize the filtered keypoints relative to the full image.
        norm_kps = normalize_keypoints(filtered_kps, W, H)
        
        out_filename = f"single_{image_id}.jpg"
        out_img_path = images_out / out_filename
        shutil.copy(str(img_path), out_img_path)
        
        label_filename = f"single_{image_id}.txt"
        label_path = labels_out / label_filename
        write_label_file(label_path, norm_kps)
        
        single_count += 1
    
    print("Pipeline complete!")
    print(f"Processed {len(img_to_anns)} images; {single_count} contain exactly one person and pass keypoint filtering.")

if __name__ == "__main__":
    # Adjust these paths as needed:
    json_path = "../datasets/annotations_trainval2017/person_keypoints_val2017.json"
    images_dir = "../datasets/coco-pose/images/val2017/val2017"  # Folder containing original images
    output_dir = "../datasets/val_subset_single"                   # Folder where the subset will be stored
    
    process_pipeline(json_path, images_dir, output_dir)


# Train:  Processed 64115 images; 19971 contain exactly one person and pass keypoint filtering.
# Val:    Processed 2693 images; 833 contain exactly one person and pass keypoint filtering.

