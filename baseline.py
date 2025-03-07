import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose, self).__init__()
        self.pre_model = torchvision.models.alexnet(pretrained=True)
        self.fc1 = nn.Linear(13*13*192, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 26)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

def train(model, train_loader, num_epochs, learning_rate, baseline=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model.pre_model.feature(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')

if __name__ == '__main__':
    model = DeepPose()