import os
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from baseline import DeepPose, KeypointDataset  # Assumes your baseline.py defines these classes

def plot_performance(performance_log_path):
    performance_log = torch.load(performance_log_path)
    epochs = [entry['epoch'] for entry in performance_log]
    train_losses = [entry['train_loss'] for entry in performance_log]
    val_losses = [entry['val_loss'] for entry in performance_log]
    val_pck = [entry['val_pck'] for entry in performance_log]
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(epochs, [v*100 for v in val_pck], marker='o', color='green', label='Val PCK (%)')
    plt.xlabel('Epoch')
    plt.ylabel('PCK (%)')
    plt.title('Validation PCK')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_last_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*_fc_checkpoint.pth"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint found in directory " + checkpoint_dir)
    # Sort checkpoints by epoch number extracted from filename.
    checkpoint_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
    return checkpoint_files[-1]

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.fc1.load_state_dict(checkpoint['fc1'])
    model.fc2.load_state_dict(checkpoint['fc2'])
    model.fc3.load_state_dict(checkpoint['fc3'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

def visualize_sample(model, image_path, label_path, transform, device):
    # Load and convert image to RGB.
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        print(f"Failed to load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Prepare image for model inference using the same transform as training.
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Inference.
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    # Constrain outputs to [0,1]
    output = torch.sigmoid(output)
    output = output.view(-1, 2).cpu().numpy()  # shape should be (13,2)
    
    # Load ground truth keypoints.
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    kp_values = parts[1:]
    kp_array = np.array(kp_values, dtype=float).reshape(-1, 3)
    gt_keypoints = kp_array[:, 0:2]  # Expected shape (13,2)

    # Resize original image to the training/inference dimensions.
    disp_size = (220, 220)
    resized_img = cv2.resize(img_rgb, disp_size)
    H, W = disp_size[1], disp_size[0]  # height, width

    # Scale normalized predicted and GT keypoints to pixel coordinates.
    for (px, py), (gx, gy) in zip(output, gt_keypoints):
        pred_x = int(px * W)
        pred_y = int(py * H)
        gt_x   = int(gx * W)
        gt_y   = int(gy * H)
        # Draw predicted keypoints in green.
        cv2.circle(resized_img, (pred_x, pred_y), 3, (0,255,0), -1)
        # Draw ground truth keypoints in blue.
        cv2.circle(resized_img, (gt_x, gt_y), 3, (255,0,0), -1)
    
    plt.figure(figsize=(6,6))
    plt.imshow(resized_img)
    plt.title("Predictions (Green) vs Ground Truth (Red)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Plot training curves.
    perf_log_path = Path("../checkpoints/performance_log.pt")
    if perf_log_path.exists():
        plot_performance(perf_log_path)
    else:
        print("Performance log not found.")

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and load last checkpoint.
    model = DeepPose()
    model.to(device)
    ckpt_dir = "../checkpoints"
    last_ckpt = "../checkpoints/epoch_2_fc_checkpoint.pth" 
    load_checkpoint(last_ckpt, model, device)

    model2 = DeepPose()
    model2.to(device)
    ckpt_dir2 = "../checkpoints"
    last_ckpt2 = "../checkpoints/epoch_50_fc_checkpoint.pth"
    load_checkpoint(last_ckpt2, model2,device)
    
    # Define transform (should match the training transform).
    transform = transforms.Compose([
        transforms.Resize((220,220)),
        transforms.ToTensor()
    ])
    
    # Use a few random samples from the dataset.
    images_dir = Path('../datasets/train_subset_single/standardized_images')
    labels_dir = Path('../datasets/train_subset_single/labels')
    image_files = list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print("No images found in", images_dir)
    else:
        # Visualize 3 random samples.
        import random
        for img_path in random.sample(image_files, 3):
            label_path = labels_dir / (img_path.stem + ".txt")
            print(f"Visualizing sample: {img_path.name}")
            visualize_sample(model, img_path, label_path, transform, device)
            visualize_sample(model2, img_path, label_path, transform, device)