import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from PIL import Image

import os
from pathlib import Path
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

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


class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose, self).__init__()
        self.pre_model = torchvision.models.alexnet(pretrained=True)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 26)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        # Extract features using the pretrained alexnet features.
        features = self.pre_model.features(x)
        # Flatten the features.
        features = features.view(features.size(0), -1)
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def compute_pck(outputs, labels, threshold=0.2):
    """
    Compute the Percentage of Correct Keypoints (PCK) for a batch.
    
    Args:
        outputs (Tensor): Model predictions of shape (batch, 26).
        labels (Tensor): Ground truth of shape (batch, 26).
        threshold (float): If the Euclidean distance is below this value (normalized), the keypoint is correct.
    Returns:
        Average PCK over the batch.
    """
    #compute hip distance label 7 and 8

    batch_size = outputs.shape[0]
    # Reshape to (batch, 13, 2)
    outputs = outputs.view(batch_size, 13, 2)
    labels = labels.view(batch_size, 13, 2)
    # Compute Euclidean distances for each keypoint.
    distances = torch.norm(outputs - labels, dim=2)  # shape: (batch, 13)
    # Consider a keypoint correct if its distance is less than the threshold.
    correct = (distances < threshold).float()
    # Average correctness per sample.
    pck_per_sample = correct.mean(dim=1)
    # Average across the batch.
    return pck_per_sample.mean().item()


def evaluate(model, data_loader, threshold=0.1):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_pck = 0.0
    count = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            labels = labels.view(labels.size(0), -1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            # Compute PCK for the current batch.
            batch_pck = compute_pck(outputs, labels, threshold)
            total_pck += batch_pck * inputs.size(0)
            count += inputs.size(0)
    avg_loss = total_loss / count
    avg_pck = total_pck / count
    print(f'Validation Loss: {avg_loss:.4f}, PCK: {avg_pck*100:.2f}%')
    return avg_loss, avg_pck


def train(model, train_loader, val_loader, num_epochs, learning_rate, pck_threshold=0.25,device='cuda'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.1, patience=5)
    # Directory to save checkpoints.
    checkpoint_dir = Path("../checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Lists to record performance (for graphing later).
    performance_log = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            labels = labels.view(labels.size(0), -1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed, Average Training Loss: {avg_train_loss:.4f}')
        
        # Evaluate on the validation set and compute PCK.
        val_loss, val_pck = evaluate(model, val_loader, threshold=pck_threshold)
        
        # Record performance for graphing.
        performance_log.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_pck': val_pck
        })
        
        # Save the fc layer parameters and performance metrics as a checkpoint.
        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}_fc_checkpoint.pth"
        torch.save(model.state_dict(), str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")
        scheduler.step(val_loss)

    
    # Optionally, save the overall performance log for graphing later.
    performance_path = checkpoint_dir / "performance_log.pt"
    torch.save(performance_log, str(performance_path))
    print(f"Saved performance log: {performance_path}")

if __name__ == '__main__':
    model = DeepPose()
    
    transform = transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.ToTensor()
    ])
    
    dataset = KeypointDataset('../datasets/train_subset_single/standardized_images',
                              '../datasets/train_subset_single/labels',
                              transform=transform,
                              max_images=500)
    
    # Split dataset: 80% training, 20% validation.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=18, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Total images: {len(dataset)}; Train: {len(train_dataset)}; Validation: {len(val_dataset)}")

    #send to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    
    train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.01, pck_threshold=0.05)



    