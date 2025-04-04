import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import time

def hard_argmax(heatmaps):
    B, J, H, W = heatmaps.shape
    # Flatten each heatmap to shape [B, J, H*W]
    flat = heatmaps.view(B, J, -1)
    # Get indices of maximum value
    indices = flat.argmax(dim=-1)  # shape [B, J]
    # Compute x and y coordinates
    x = indices % W
    y = indices // W
    coords = torch.stack((x.float(), y.float()), dim=-1)
    return coords

# Model Architecture
class ClosedPose(nn.Module):
    def __init__(self, num_joints=13):
        super(ClosedPose, self).__init__()
        self.name = "pose"
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, 2, 0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(2048, 1536, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(1536, 1024, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(128, num_joints, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(num_joints, num_joints, kernel_size=1),
        )
    
    def forward(self, x):
        # features = self.feature_extractor(x)
        heatmaps = self.decoder(x)
        heatmaps = torch.sigmoid(heatmaps)
        return heatmaps

# Soft Argmax with Threshold
def soft_argmax(heatmaps, threshold=None):
    """
    Computes the soft-argmax to extract (x, y) coordinates from heatmaps.
    
    Parameters:
      heatmaps: tensor of shape [B, num_joints, H, W].
      threshold: if provided, zero out probabilities below this value (optional).
    
    Returns:
      coords: tensor of shape [B, num_joints, 2] with estimated x, y coordinates.
    """
    batch_size, num_joints, H, W = heatmaps.shape
    heatmaps = heatmaps.view(batch_size, num_joints, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)
    
    if threshold is not None:
        heatmaps = torch.where(heatmaps > threshold, heatmaps, torch.zeros_like(heatmaps))
        # Renormalize after thresholding
        heatmaps = heatmaps / (heatmaps.sum(dim=-1, keepdim=True) + 1e-6)
    
    # Create coordinate grids.
    x_grid = torch.linspace(0, W - 1, W, device=heatmaps.device).repeat(H, 1).view(H * W)
    y_grid = torch.linspace(0, H - 1, H, device=heatmaps.device).repeat(W, 1).t().contiguous().view(H * W)
    
    x = torch.sum(heatmaps * x_grid, dim=-1)
    y = torch.sum(heatmaps * y_grid, dim=-1)
    coords = torch.stack((x, y), dim=-1)
    return coords

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
        embedding_tensor = embedding_tensor.squeeze(1)

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
        kp_array *= 224
        label_tensor = torch.tensor(kp_array, dtype=torch.float32)
        
        return embedding_tensor, label_tensor

# Utility Functions
def generate_heatmaps_batch(keypoints, H=224, W=224, sigma=4):
    """
    Vectorized generation of Gaussian heatmaps for a batch of keypoints.
    keypoints: tensor of shape [B, num_joints, 2] in pixel space
    Returns: tensor of shape [B, num_joints, H, W]
    If a keypoint is (0,0) (assumed missing/invisible), its heatmap will be all zeros.
    """
    B, J, _ = keypoints.shape
    device = keypoints.device
    # Create coordinate grid once.
    x_lin = torch.linspace(0, W - 1, W, device=device)
    y_lin = torch.linspace(0, H - 1, H, device=device)
    y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing="ij")  # shape: [H, W]
    # Expand grid to match batch & joints.
    # Final shapes: [B, J, H, W]
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)
    
    # Expand keypoints from shape [B, J, 2] to [B, J, 1, 1]
    kp_exp = keypoints.unsqueeze(-1).unsqueeze(-1)
    kp_x = kp_exp[:, :, 0, :, :]  # shape [B, J, 1, 1]
    kp_y = kp_exp[:, :, 1, :, :]
    
    # Compute squared distance from the keypoint location
    dist_sq = (x_grid - kp_x)**2 + (y_grid - kp_y)**2
    # Compute Gaussian heatmaps
    heatmaps = torch.exp(-dist_sq / (2 * sigma**2))
    
    # If a keypoint is (0,0) (assuming missing) then set its heatmap to zero.
    mask_missing = (keypoints.abs().sum(dim=-1) == 0).unsqueeze(-1).unsqueeze(-1)
    heatmaps = heatmaps * (1 - mask_missing.float())
    return heatmaps



# Training and Evaluation Functions
class KeypointDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, max_images=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_paths = [p for p in self.images_dir.glob("*.jpg") if (self.labels_dir / f"{p.stem}.txt").exists()]
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = Image.fromarray(img)
            image_tensor = self.transform(img)
        
        label_path = self.labels_dir / (image_path.stem + ".txt")
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        parts = line.split()
        kp_values = parts[1:]
        kp_array = np.array(kp_values, dtype=float).reshape(-1, 3)[:, 0:2]
        kp_array *= 224
        label_tensor = torch.tensor(kp_array, dtype=torch.float32)
        return image_tensor, label_tensor

# Focal Loss
def weighted_mse_loss(pred, target, alpha=55.0, beta=5.0, threshold=0.1):
    """
    Compute a weighted mean squared error loss.
    Pixels with ground truth value greater than 'threshold' are considered foreground and weighted by alpha.
    Background pixels are weighted by beta.
    """
    # Create a weight tensor matching the target shape
    weights = torch.where(target > threshold,
                          torch.tensor(alpha, device=target.device),
                          torch.tensor(beta, device=target.device))
    loss = weights * (pred - target) ** 2
    #loss = (pred-target) ** 2
    return loss.mean()

# Training Function
def train(model, train_loader, val_loader, batch_size=30, learning_rate=0.00001, num_epochs=150, device='cuda'):
    torch.manual_seed(420)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    H, W = 224, 224
    data = [0] * num_epochs
    
    for epoch in range(num_epochs):
        s = time.time()
        model.train()
        running_loss = 0.0
        running_train_pck = 0.0
        train_count = 0
        
        for images, labels in train_loader:
            # print(images.shape, labels.shape)
            images, labels = images.to(device), labels.to(device)  # labels: [B, 13, 2]
            optimizer.zero_grad()
            images = images.squeeze(1)
            heatmap_outputs = model(images)
            heatmap_outputs = F.interpolate(heatmap_outputs, size=(H, W), mode='bilinear', align_corners=False)
            # Batch generate ground truth heatmaps (shape: [B, 13, H, W])
            gt_heatmaps = generate_heatmaps_batch(labels, H=H, W=W, sigma=4)
            
            loss = weighted_mse_loss(heatmap_outputs, gt_heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
            
            # Compute training PCK on this batch
            batch_pck = compute_pck(heatmap_outputs, labels)
            bsize = images.size(0)
            running_train_pck += batch_pck * bsize
            train_count += bsize
        
        avg_train_loss = running_loss / len(train_loader)
        avg_train_pck = running_train_pck / train_count
        
        model.eval()
        val_running_loss = 0.0
        running_val_pck = 0.0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.squeeze(1)
                heatmap_outputs = model(images)
                heatmap_outputs = F.interpolate(heatmap_outputs, size=(H, W), mode='bilinear', align_corners=False)
                gt_heatmaps = generate_heatmaps_batch(labels, H=H, W=W, sigma=4)
                loss = weighted_mse_loss(heatmap_outputs, gt_heatmaps)
                val_running_loss += loss.item()
                
                batch_pck = compute_pck(heatmap_outputs, labels)
                bsize = images.size(0)
                running_val_pck += batch_pck * bsize
                val_count += bsize
                
        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_pck = running_val_pck / val_count
        scheduler.step(avg_val_loss)
        e = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} took {e-s:.2f}s")
        
        print(f"lr={learning_rate}, batch size={batch_size}"
              f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train PCK: {avg_train_pck*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val PCK: {avg_val_pck*100:.2f}%")

        torch.save(model.state_dict(), "../checkpoints/" + "model2_" + str(learning_rate) + "_" + str(batch_size) + "_" + str(epoch) + ".pth")
        data[epoch] = [epoch, avg_train_loss, avg_train_pck, avg_val_loss, avg_val_pck]

        # if (epoch + 1) % 10 == 0:
        #     with torch.no_grad():
        #         images, labels = next(iter(val_loader))
        #         images = images.to(device)
        #         images = images.squeeze(1)
        #         pred_heatmaps = model(images)
        #         pred_heatmaps = F.interpolate(pred_heatmaps, size=(H, W), mode='bilinear', align_corners=False)
        #         gt_heatmaps = generate_heatmaps_batch(labels.to(device), H=H, W=W, sigma=4)
        #         plot_heatmaps(images, gt_heatmaps, pred_heatmaps)
    data = np.asarray(data)
    np.savetxt("../stats/" + "model2_" + str(learning_rate) + "_" + str(batch_size) + "_" + str(epoch) + ".csv", data, header="epoch, train loss, train pck, val loss, val pck", delimiter=",")
    return model

def evaluate(model, data_loader, device='cuda', threshold=20):
    model.eval()
    total_loss = 0.0
    total_pck = 0.0
    count = 0
    H, W = 224, 224
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=False)
            gt_heatmaps = generate_heatmaps_batch(labels, H=H, W=W, sigma=4)
            loss = weighted_mse_loss(outputs, gt_heatmaps)
            total_loss += loss.item() * inputs.size(0)
            batch_pck = compute_pck(outputs, labels, threshold)
            total_pck += batch_pck * inputs.size(0)
            count += inputs.size(0)
    avg_loss = total_loss / count
    avg_pck = total_pck / count
    print(f'Validation Loss: {avg_loss:.4f}, PCK: {avg_pck*100:.2f}%')
    return avg_loss, avg_pck

# PCK Evaluation
def compute_pck(outputs, labels, threshold=5):
    """
    Compute PCK only for keypoints whose labels are not [0, 0].
    outputs: predicted heatmaps used with soft_argmax to get coordinates (shape: [B, 13, H, W])
    labels: ground truth keypoints (shape: [B, 13, 2])
    threshold: distance threshold (in pixels)
    """
    coords = hard_argmax(outputs)  # shape: [B, 13, 2]
    batch_size = coords.shape[0]
    labels = labels.view(batch_size, -1, 2)  # [B, 13, 2]
    
    # Create a mask for valid keypoints (i.e., not both coordinates equal to 0)
    valid_mask = ~((labels == 0).all(dim=2))  # shape: [B, 13], True if keypoint is valid
    
    # Compute Euclidean distances per keypoint
    distances = torch.norm(coords - labels, dim=2)  # shape: [B, 13]
    
    # For each keypoint, if distance < threshold, mark it correct (only if valid)
    correct = ((distances < threshold).float() * valid_mask.float())  # shape: [B, 13]
    
    # For each sample, count number of valid keypoints (avoid division by zero)
    valid_counts = valid_mask.sum(dim=1).float()  # shape: [B]
    
    per_sample_pck = []
    for i in range(batch_size):
        if valid_counts[i] > 0:
            pck = correct[i].sum() / valid_counts[i]
        else:
            pck = torch.tensor(0.0, device=correct.device)
        per_sample_pck.append(pck)
    per_sample_pck = torch.stack(per_sample_pck)
    return per_sample_pck.mean().item()
# Visualization
def plot_heatmaps(images, gt_heatmaps, pred_heatmaps, num_samples=2):
    for i in range(min(num_samples, images.size(0))):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(gt_heatmaps[i, 0].cpu().numpy())
        plt.title("Ground Truth Heatmap")
        plt.subplot(1, 3, 3)
        plt.imshow(pred_heatmaps[i, 0].cpu().numpy())
        plt.title("Predicted Heatmap")
        plt.show()

if __name__ == "__main__":
    torch.manual_seed(420)
    val_data = EmbedKeypointDataset('../datasets/val_subset_single/embeddings', '../datasets/val_subset_single/labels')
    train_data = EmbedKeypointDataset('../datasets/train_subset_single/embeddings', '../datasets/train_subset_single/labels')
    for j in [10, 300, 100, 30]:
        train_loader = DataLoader(train_data, batch_size=j)
        val_loader = DataLoader(val_data, batch_size=j)
        for i in [0.00001, 0.0001, 0.00003, 0.0003]:
            model = ClosedPose()
            if torch.cuda.is_available():
                model = model.cuda()
                print("Using GPU")
            train(model, train_loader, val_loader, learning_rate=i, batch_size=j, num_epochs=70)
    