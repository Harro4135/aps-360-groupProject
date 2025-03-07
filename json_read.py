import json

def count_unique_images(json_path):
    """
    Load COCO person_keypoints JSON file and return the number of unique images.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    unique_image_ids = {img['id'] for img in data.get('images', [])}
    return len(unique_image_ids)

def count_human_images(json_path):
    """
    Load COCO person_keypoints JSON file and return the number of unique images that contain 
    at least one human annotation. Assumes that human annotations have a category_id == 1.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    human_image_ids = {ann['image_id'] for ann in data.get('annotations', []) if ann.get('category_id') == 1}
    return len(human_image_ids)

def count_human_images_with_usable_annotations(json_path):
    """
    Load COCO person_keypoints JSON file and return the number of unique images that contain 
    at least one human annotation with usable keypoints.
    An annotation is considered usable if:
      - The 'keypoints' list length is a multiple of 3.
      - All keypoint visibility flags (every third item in the keypoints list) are either 1 or 2.
    Assumes that human annotations have a category_id == 1.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    usable_image_ids = set()
    for ann in data.get('annotations', []):
        if ann.get('category_id') != 1:
            continue
        
        keypoints = ann.get('keypoints', [])
        if len(keypoints) % 3 != 0:
            continue
        
        # Check all visibility flags
        if all(keypoints[3*i+2] in (1, 2) for i in range(len(keypoints)//3)):
            usable_image_ids.add(ann['image_id'])
            
    return len(usable_image_ids)

if __name__ == "__main__":
    json_path = "person_keypoints_val2017.json"
    num_unique_images = count_unique_images(json_path)
    num_human_images = count_human_images(json_path)
    num_human_images_usable = count_human_images_with_usable_annotations(json_path)
    
    print(f"Found {num_unique_images} unique images in total.")
    print(f"Of these, {num_human_images} images have human annotations.")
    print(f"Of the human images, {num_human_images_usable} have usable annotations.")