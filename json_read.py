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

def process_pipeline(json_path, images_dir, output_dir, padding=0.1, min_crop_w=30, min_crop_h=30):
    """
    Process images from a COCO person_keypoints JSON.
    
    For each image:
      - If only one person is annotated, copy the full image and generate a label file using normalized
        keypoints (relative to the full image).
      - If more than one person is annotated, for each person:
          * Skip the annotation if "iscrowd" is set to 1.
          * Crop the region using the annotation bbox (with optional padding).
          * If the resulting crop's width or height is below the minimum size, skip that annotation.
          * Otherwise, adjust keypoints (normalize relative to the crop), and write the crop and its label.
    
    Args:
      json_path (str): Path to the COCO annotation JSON file.
      images_dir (str or Path): Directory containing the source images.
      output_dir (str or Path): Directory to save the prepared test subset.
      padding (float): Fractional padding to add around each bbox when cropping.
      min_crop_w (int): Minimum width of the cropped image.
      min_crop_h (int): Minimum height of the cropped image.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Build a mapping from image_id to image metadata.
    image_map = {img['id']: img for img in data.get('images', [])}
    
    # Group annotations by image_id for category_id == 1.
    img_to_anns = {}
    for ann in data.get('annotations', []):
        if ann.get('category_id') == 1:
            img_to_anns.setdefault(ann['image_id'], []).append(ann)
    
    crop_id = 1
    single_count = 0
    multi_count = 0

    for image_id, anns in img_to_anns.items():
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
        
        if len(anns) == 1:
            # Single-person image: process if annotation is not a crowd.
            ann = anns[0]
            if ann.get('iscrowd', 0) == 1:
                continue  # Skip if "iscrowd" is set.
            keypoints = ann.get('keypoints', [])
            if len(keypoints) % 3 != 0:
                continue  # Skip if keypoints not valid.
            norm_kps = normalize_keypoints(keypoints, W, H)
            
            out_filename = f"single_{image_id}.jpg"
            out_img_path = images_out / out_filename
            shutil.copy(str(img_path), out_img_path)
            
            label_filename = f"single_{image_id}.txt"
            label_path = labels_out / label_filename
            write_label_file(label_path, norm_kps)
            single_count += 1
            
        else:
            # Multi-person image: crop each person's bbox if annotation is not a crowd.
            for ann in anns:
                if ann.get('iscrowd', 0) == 1:
                    continue  # Skip crowd annotations.
                if 'bbox' not in ann or 'keypoints' not in ann:
                    continue
                bbox = ann['bbox']  # Format: [x, y, w, h]
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                pad_w = int(w * padding)
                pad_h = int(h * padding)
                x1 = max(int(x) - pad_w, 0)
                y1 = max(int(y) - pad_h, 0)
                x2 = min(int(x + w) + pad_w, W)
                y2 = min(int(y + h) + pad_h, H)
                
                crop = image[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                # Skip the crop if it doesn't meet the minimum dimensions.
                if crop_w < min_crop_w or crop_h < min_crop_h:
                    continue
                
                crop_filename = f"crop_{crop_id}.jpg"
                crop_path = images_out / crop_filename
                cv2.imwrite(str(crop_path), crop)
                
                orig_kps = ann.get('keypoints', [])
                num_kps = len(orig_kps) // 3
                adjusted_kps = []
                for i in range(num_kps):
                    kp_x = orig_kps[3*i] - x1
                    kp_y = orig_kps[3*i+1] - y1
                    vis = orig_kps[3*i+2]
                    norm_x = kp_x / crop_w
                    norm_y = kp_y / crop_h
                    adjusted_kps.extend([norm_x, norm_y, vis])
                
                label_filename = f"crop_{crop_id}.txt"
                label_path = labels_out / label_filename
                write_label_file(label_path, adjusted_kps)
                
                crop_id += 1
                multi_count += 1
    
    print("Pipeline complete!")
    print(f"Processed {len(img_to_anns)} images: {single_count} single-person images and {multi_count} person crops from multi-person images.")

if __name__ == "__main__":
    # Adjust these paths as needed:
    json_path = "../datasets/annotations_trainval2017/person_keypoints_train2017.json"
    images_dir = "../datasets/coco-pose/images/train2017/train2017"  # Folder containing original images
    output_dir = "../datasets/train_subset_crop"                       # Folder where the test subset will be stored
    
    process_pipeline(json_path, images_dir, output_dir, padding=0.1)


    #Processed 64115 images: 24832 single-person images and 144444 person crops from multi-person images.

    #Processed 2693 images: 1045 single-person images and 6196 person crops from multi-person images.