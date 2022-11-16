# This file defining the model was taken as-is from https://github.com/Azure/medical-imaging/blob/main/federated-learning/pneumonia-federated/custom/pneumonia_network.py.
import torch
import torch.nn as nn
import torch.nn.functional as F


class PneumoniaNetwork(nn.Module):
    def __init__(self):
        super(PneumoniaNetwork, self).__init__()
        dropout = 0.2

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(28 * 28 * 128, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 224 x 224 x 32
        x = F.max_pool2d(x, 2, 2)  # 112 x 112 x 32
        x = F.relu(self.conv2(x))  # 112 x 112 x 64
        x = F.max_pool2d(x, 2, 2)  # 56 x 56 x 64
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))  # 56 x 56 x 128
        x = F.max_pool2d(x, 2, 2)  # 28 x 28 x 128
        x = self.dropout2(x)
        x = x.view(-1, 28 * 28 * 128)  # 100.352
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
