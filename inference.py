import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from model_heatmap_embeds import ClosedPose
from model_heatmap_embeds import hard_argmax
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

JOINT_NAMES = [
    "Head", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
    "Right Knee", "Left Ankle", "Right Ankle"
]

# Define skeleton connections (parent-child pairs) for visualization
# This assumes a standard human pose structure; adjust if your dataset uses a different ordering
SKELETON = [
    (0, 1), (0, 2),  # Head to shoulders
    (1, 3), (3, 5),  # Left shoulder to elbow to wrist
    (2, 4), (4, 6),  # Right shoulder to elbow to wrist
    (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),          # Left hip to right hip
    (7, 9), (9, 11), # Left hip to knee to ankle
    (8, 10), (10, 12) # Right hip to knee to ankle
]
model = ClosedPose()
path = torch.load("../checkpoints/model1_0.0003_100_41.pth")
model.load_state_dict(path)
resnet50 = models.resnet50(weights='IMAGENET1K_V1')
resnet50 = resnet50.eval()

feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])

model = model.eval()
feature_extractor = feature_extractor.eval()

if torch.cuda.is_available():
    model = model.cuda()
    feature_extractor = feature_extractor.cuda()

num = "498807"

image_path = "../datasets/val_subset_single/standardized_images/single_"+num+".jpg"

label_path = "../datasets/val_subset_single/labels/single_"+num+".txt"

image_path = "../datasets/IMG_1498.jpg"

img = cv2.imread(str(image_path))
if img is None:
    raise RuntimeError(f"Failed to load image: {image_path}")
# Convert from BGR to RGB.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Resize to 224x224 (or whatever size your model expects).
img = cv2.resize(img, (224, 224))
# Convert to tensor and normalize.
img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
# Add batch dimension.
img_tensor = img_tensor.unsqueeze(0)
if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()


out = model(feature_extractor(img_tensor))

def plot_keypoints_and_heatmaps(images, labels, pred_heatmaps, num_samples=2):
    """
    For each sample in the batch:
      - Ground truth keypoints are extracted from `labels` (assumed normalized in [0,1]) and scaled
        to pixel space.
      - Predicted keypoints are extracted from `pred_heatmaps` by applying a softmax over each joint's heatmap.
      - A panel is created that shows:
          1. The input image with ground truth keypoints.
          2. The input image with predicted keypoints.
          3. A grid of the predicted heatmaps with softmax probabilities overlaid.
    
    Parameters:
      images      : Tensor of shape [B, 3, H, W] (normalized as in training).
      labels      : Tensor of shape [B, num_joints, 2] containing normalized keypoint coordinates.
      pred_heatmaps: Tensor of shape [B, num_joints, h, w] predicted by the model.
      num_samples : Number of images to display.
    """
    pred_heatmaps = F.interpolate(pred_heatmaps, size=(224, 224), mode='bilinear', align_corners=False)
    num_joints = labels.shape[1]
    for i in range(min(num_samples, images.size(0))):
        # ---- Denormalize and prepare image ----
        img = images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        print(labels)
        
        # ---- Ground truth keypoints ----
        # Labels are assumed normalized in [0,1]; scale to [0, H-1] (224 in our case)
        gt_coords = labels[i].cpu().numpy() 
        
        # ---- Predicted keypoints ----
        # Apply soft_argmax on each joint's predicted heatmap.
        # Assume soft_argmax() takes a tensor of shape [B, num_joints, H, W] and returns
        # expected coordinates in pixel space.

        pred_coords = hard_argmax(pred_heatmaps[i].unsqueeze(0))[0].cpu().numpy()  # shape: [num_joints, 2]
        
        # ---- Plot overlays ----
        plt.figure(figsize=(18,6))
        
        # Panel 1: Ground Truth Keypoints Overlay
        plt.subplot(1,3,1)
        plt.imshow(img)
        print("GT coords: ", gt_coords)
        print(len(gt_coords))
        for j, (x, y) in enumerate(gt_coords):
            plt.scatter(x, y, color='red', s=60)
            plt.text(x + 3, y, JOINT_NAMES[j], color='red', fontsize=8)
        for (j1, j2) in SKELETON:
            print("Skeleton: ", j1, j2)
            x1, y1 = gt_coords[j1]
            x2, y2 = gt_coords[j2]
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
        plt.title("Ground Truth Keypoints")
        
        # Panel 2: Predicted Keypoints Overlay
        plt.subplot(1,3,2)
        plt.imshow(img)
        for j, (x, y) in enumerate(pred_coords):
            plt.scatter(x, y, color='green', s=60)
            plt.text(x + 3, y, JOINT_NAMES[j], color='green', fontsize=8)
        for (j1, j2) in SKELETON:
            x1, y1 = pred_coords[j1]
            x2, y2 = pred_coords[j2]
            plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1)
        plt.title("Predicted Keypoints")
        
        # Panel 3: Display each predicted heatmap after softmax
        # Determine grid shape for plotting all num_joints channels.
        ncols = int(np.ceil(np.sqrt(num_joints)))
        nrows = int(np.ceil(num_joints / ncols))
        fig_heat, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
        axs = np.array(axs).reshape(-1)
        pred_heatmaps_i = pred_heatmaps[i]  # shape: [num_joints, h, w]
        # cv2.imshow("", img)
        for j in range(num_joints):
            heatmap = pred_heatmaps_i[j]
            # Apply softmax over the flattened heatmap channel
            flat = heatmap.view(-1)
            probs = torch.softmax(flat, dim=-1).view(heatmap.shape)
            axs[j].imshow(img)
            axs[j].imshow(probs.cpu().detach().numpy(), cmap='jet', alpha=0.5)
            axs[j].set_title(JOINT_NAMES[j])
            axs[j].axis('off')
        # Turn off extra subplots, if any.
        for ax in axs[num_joints:]:
            ax.axis('off')
        plt.tight_layout()
        
        plt.show()

images = cv2.imread(str(image_path))

with open(label_path, 'r') as f:
    line = f.readline().strip()
parts = line.split()
kp_values = parts[1:]
kp_array = np.array(kp_values, dtype=float).reshape(-1, 3)[:, 0:2]
kp_array *= 224
label_tensor = torch.tensor(kp_array, dtype=torch.float32)

label_tensor = label_tensor.unsqueeze(0)
# img_tensor = img_tensor.unsqueeze(1)
print("Label tensor: ", label_tensor.shape)
label_tensor = torch.zeros(label_tensor.shape)

plot_keypoints_and_heatmaps(img_tensor, label_tensor, out, num_samples=1)