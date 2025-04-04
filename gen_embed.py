import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import torchvision.models as models # for ResNet

from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path

resnet50 = models.resnet50(weights='IMAGENET1K_V1')

feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])

# create embeddings for training

class KeypointDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, max_images=None):
        """
        Args:
            images_dir (str or Path): Directory containing image files.
            labels_dir (str or Path): Directory containing label files.
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
        # Load image.
        image_path = self.image_paths[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        # Convert from BGR to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations if provided (expects a PIL Image).
        if self.transform:
            img = Image.fromarray(img)
            image_tensor = self.transform(img)
        else:
            image_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        # Load keypoint labels.
        label_path = self.labels_dir / (image_path.stem + ".txt")
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        parts = line.split()
        # Skip the first token (dummy class id).
        kp_values = parts[1:]
        # Convert to numpy array and reshape to (num_keypoints, 3)
        kp_array = np.array(kp_values, dtype=float).reshape(-1, 3)
        # Remove first entry of each keypoint (the class id)
        kp_array = kp_array[:, 0:2]
        # Convert to tensor.
        label_tensor = torch.tensor(kp_array, dtype=torch.float32)
        # Flatten the label tensor to match the network's output (batch x 26)
        label_tensor = label_tensor.view(-1)
        
        return image_tensor, label_tensor

def create_embeddings(feature_extractor):
    '''create and save embeddings for images in the image folder all individually'''
    source_dir = '../datasets/val_subset_single/standardized_images'
    target_dir = Path('../datasets/val_subset_single/embeddings')
    target_dir.mkdir(parents=True, exist_ok=True)
    image_paths = Path(source_dir).glob("*.jpg")
    feature_extractor.eval()
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()
    # print(f"Found {len(list(image_paths))} images in {source_dir}.")
    # print(list(image_paths))
    for image_path in list(image_paths):
        print(f"Processing {image_path}...")
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        # Convert from BGR to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to tensor and normalize.
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        # Add batch dimension.
        img_tensor = img_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Extract features.
        with torch.no_grad():
            feature_map = feature_extractor(img_tensor)

        # feature map to cpu
        feature_map = feature_map.cpu()
        # Save the feature map as a .npy file.
        np.save(target_dir / f"{image_path.stem}.npy", feature_map.numpy())

class EmbedKeypointDataset(Dataset):
    def __init__(self, embeds_dir, labels_dir, transform=None, max_images=None):
        """
        Args:
            images_dir (str or Path): Directory containing image files.
            labels_dir (str or Path): Directory containing label files.
            transform (callable, optional): A torchvision.transforms transformation to apply.
            max_images (int, optional): If set, cap the dataset to at most max_images images.
        """
        self.embeds_dir = Path(embeds_dir)
        self.labels_dir = Path(labels_dir)
        # Only include images that have a matching label file.
        self.embed_paths = [p for p in self.embeds_dir.glob("*.npy")
                            if (self.labels_dir / f"{p.stem}.txt").exists()]
        if max_images is not None:
            self.embed_paths = self.embed_paths[:max_images]
        self.transform = transform
    
    def __len__(self):
        return len(self.embed_paths)

    def __getitem__(self, idx):
        embed_path = self.embed_paths[idx]
        # Load the embedding from the .npy file.
        embedding = np.load(embed_path)
        if self.transform:
            embedding = self.transform(embedding)
        embedding_tensor = torch.from_numpy(embedding)

        # Load the keypoint labels.
        label_path = self.labels_dir / f"{embed_path.stem}.txt"
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        parts = line.split()
        # Skip the first token (dummy class id).
        kp_values = parts[1:]
        # Convert to numpy array and reshape to (num_keypoints, 3)
        kp_array = np.array(kp_values, dtype=float).reshape(-1, 3)
        # Remove the class id (first value) from each keypoint.
        kp_array = kp_array[:, :2]
        label_tensor = torch.tensor(kp_array, dtype=torch.float32).view(-1)
        
        return embedding_tensor, label_tensor
        

if __name__ == "__main__":
    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = KeypointDataset('../datasets/val_subset_single/standardized_images',
                                  '../datasets/val_subset_single/labels',
                                  transform=transform,
                                  max_images=500)    
    # Create embeddings
    create_embeddings(feature_extractor)
    print("Embeddings created and saved successfully.")