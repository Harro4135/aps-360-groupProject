import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class KeypointDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, max_images=None):
        """
        Args:
            images_dir (str or Path): Directory containing the image files.
            labels_dir (str or Path): Directory containing the label files.
            transform (callable, optional): A torchvision.transforms transformation to apply.
            max_images (int, optional): If set, cap the dataset to at most max_images images.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        # Only include images that have a matching label file.
        self.image_paths = [p for p in self.images_dir.glob("*.jpg")
                            if (self.labels_dir / f"{p.stem}.txt").exists()]
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations if provided (expects a PIL image)
        if self.transform:
            img = Image.fromarray(img)
            image_tensor = self.transform(img)
        else:
            # Fallback: convert to tensor using default processing
            image_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        # Load keypoint labels from file
        label_path = self.labels_dir / (image_path.stem + ".txt")
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        parts = line.split()
        # First token is a dummy class; skip it.
        kp_values = parts[1:]
        # Convert to numpy array and reshape it to (num_keypoints, 3)
        kp_array = np.array(kp_values, dtype=float).reshape(-1, 3)
        # remove first entry of each keypoint (class id)
        kp_array = kp_array[:, 1:]
        # Convert to tensor.
        label_tensor = torch.tensor(kp_array, dtype=torch.float32)

        return image_tensor, label_tensor

if __name__ == "__main__":
    # Set directories; adjust as necessary.
    images_dir = "../datasets/train_subset_single/standardized_images"
    labels_dir = "../datasets/train_subset_single/labels"
    
    # Cap the dataset to a maximum number of images if desired (e.g., 500)
    max_images = 5000
    
    # Define transforms (example uses ToTensor, but you can add resizing, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Adjust with Resize((224,224)) if needed.
    ])
    
    # Create the full dataset.
    full_dataset = KeypointDataset(images_dir, labels_dir, transform=transform, max_images=max_images)
    
    # Create a split. For example, 80% for "train" (evaluation) and 20% for "val".
    dataset_size = len(full_dataset)
    train_size = int((4/5) * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Total images: {dataset_size}; train split: {len(train_dataset)}; test split: {len(test_dataset)}")
    
    # Create DataLoaders for each split.
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    

    for batch_images, batch_labels in train_loader:
        print("Train Batch images shape:", batch_images.shape)    
        print("Train Batch keypoints shape:", batch_labels.shape) 
        break
    

    for batch_images, batch_labels in test_loader:
        print("Test Batch images shape:", batch_images.shape)
        print("Test Batch keypoints shape:", batch_labels.shape)
        break


