import model_heatmap_embeds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models # for ResNet
from torch.utils.data import Dataset

model_path = "../checkpoints/model1_0.0003_100_41.pth"
model = model_heatmap_embeds.ClosedPose()
path = torch.load(model_path)
model.load_state_dict(path)
resnet50 = models.resnet50(weights='IMAGENET1K_V1')
resnet50 = resnet50.eval()
model = model.eval()
feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])
feature_extractor = feature_extractor.eval()
if torch.cuda.is_available():
    model = model.cuda()
    feature_extractor = feature_extractor.cuda()

test_data = model_heatmap_embeds.EmbedKeypointDataset("../datasets/val_subset_single/embeddings",
                                                  "../datasets/val_subset_single/labels")
split = int(len(test_data) * 0.3)
test_data = torch.utils.data.random_split(test_data, [split, len(test_data) - split], generator=torch.Generator().manual_seed(42))[0]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=split)
print("Test data size: ", len(test_data))
print("Test loader size: ", len(test_loader))
for embed, label in test_loader:
    embed = embed.cuda()
    label = label.cuda()
    with torch.no_grad():
        embed = embed.squeeze(1)
        outputs = model(embed)
        print(outputs.shape)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        pck20 = model_heatmap_embeds.compute_pck(outputs, label, threshold=20)
        pck15 = model_heatmap_embeds.compute_pck(outputs, label, threshold=15)
        pck10 = model_heatmap_embeds.compute_pck(outputs, label, threshold=10)
        pck5 = model_heatmap_embeds.compute_pck(outputs, label, threshold=5)
print("Threshold 20: ", pck20)
print("Threshold 15: ", pck15)
print("Threshold 10: ", pck10)
print("Threshold 5: ", pck5)

# # train pck
# train_data = model_heatmap_embeds.EmbedKeypointDataset("../datasets/train_subset_single/embeddings",
#                                                   "../datasets/train_subset_single/labels")
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=300)
# print("Train data size: ", len(train_data))
# print("Train loader size: ", len(train_loader))
# for embed, label in train_loader:
#     embed = embed.cuda()
#     label = label.cuda()
#     with torch.no_grad():
#         embed = embed.squeeze(1)
#         outputs = model(embed)
#         print(outputs.shape)
#         outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
#         pck20 += model_heatmap_embeds.compute_pck(outputs, label, threshold=20) * embed.shape[0]
#         pck15 += model_heatmap_embeds.compute_pck(outputs, label, threshold=15) * embed.shape[0]
#         pck10 += model_heatmap_embeds.compute_pck(outputs, label, threshold=10) * embed.shape[0]
#         pck5 += model_heatmap_embeds.compute_pck(outputs, label, threshold=5) * embed.shape[0]
# pck20 = pck20 / len(train_data)
# pck15 = pck15 / len(train_data)
# pck10 = pck10 / len(train_data)
# pck5 = pck5 / len(train_data)
# print("Train Threshold 20: ", pck20)
# print("Train Threshold 15: ", pck15)
# print("Train Threshold 10: ", pck10)
# print("Train Threshold 5: ", pck5)

# # val pck
# val_data = model_heatmap_embeds.EmbedKeypointDataset("../datasets/val_subset_single/embeddings",
#                                                   "../datasets/val_subset_single/labels")
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
# print("Val data size: ", len(val_data))
# print("Val loader size: ", len(val_loader))
# for embed, label in val_loader:
#     embed = embed.cuda()
#     label = label.cuda()
#     with torch.no_grad():
#         embed = embed.squeeze(1)
#         outputs = model(embed)
#         print(outputs.shape)
#         outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
#         pck20 = model_heatmap_embeds.compute_pck(outputs, label, threshold=20)
#         pck15 = model_heatmap_embeds.compute_pck(outputs, label, threshold=15)
#         pck10 = model_heatmap_embeds.compute_pck(outputs, label, threshold=10)
#         pck5 = model_heatmap_embeds.compute_pck(outputs, label, threshold=5)
# print("Val Threshold 20: ", pck20)
# print("Val Threshold 15: ", pck15)
# print("Val Threshold 10: ", pck10)
# print("Val Threshold 5: ", pck5)