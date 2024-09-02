import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class deadliftRegressionModel(nn.Module):
    def __init__(self, l1_lambda=0.01):
        super(deadliftRegressionModel, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * 8, 128)  # 8 here is the number of features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output is a single score

        # Define batch normalization
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        # L1 regularization weight
        self.l1_lambda = l1_lambda

    def forward(self, x):
        # Apply convolutional layers with activation functions and batch normalization
        x = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 8)  # 8 here is the number of features

        # Apply fully connected layers with activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def l1_regularization_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_lambda * l1_loss
