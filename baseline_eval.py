import baseline
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import model_heatmap_embeds


model = baseline.DeepPose()
path = torch.load("../checkpoints/epoch_65_fc_checkpoint.pth")
model.load_state_dict(path)

if torch.cuda.is_available():
    model = model.cuda()
model = model.eval()

test_data = baseline.KeypointDataset("../datasets/val_subset_single/standardized_images",
                                     "../datasets/val_subset_single/labels")
split = int(len(test_data) * 0.3)
test_data = torch.utils.data.random_split(test_data, [split, len(test_data) - split], generator=torch.Generator().manual_seed(42))[0]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
pck5 = 0
pck10 = 0
pck15 = 0
pck20 = 0
for embed, label in test_loader:
    embed = embed.numpy()
    embed = embed[:, :, :221, :221]
    embed = torch.tensor(embed)
    embed = embed.cuda()
    label = label.cuda()
    with torch.no_grad():
        # embed = embed.squeeze(1)
        print(embed.shape)
        outputs = model(embed)
        print(outputs.shape)
        pck20 += baseline.compute_pck(outputs, label, threshold=20/220)
        pck15 += baseline.compute_pck(outputs, label, threshold=15/220)
        pck10 += baseline.compute_pck(outputs, label, threshold=10/220)
        pck5 += baseline.compute_pck(outputs, label, threshold=5/220)
pck20 = pck20 / len(test_data)
pck15 = pck15 / len(test_data)
pck10 = pck10 / len(test_data)
pck5 = pck5 / len(test_data)
print("Threshold 20: ", pck20)
print("Threshold 15: ", pck15)
print("Threshold 10: ", pck10)
print("Threshold 5: ", pck5)