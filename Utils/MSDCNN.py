# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:39:59 2025

@author: jpeeples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MSDCNN(nn.Module):
    def __init__(self, num_channels=3, img_size=128, num_classes=2):
        super(MSDCNN, self).__init__()

        # Image size (Height, Width) (assume square image)
        self.img_size = (img_size,img_size)

        # First Convolutional Block (6 filters, 5x5 kernel, dilation 1, 2, 3)
        self.conv1a = nn.Conv2d(num_channels, 6, kernel_size=5, padding=2, dilation=1)
        self.conv1b = nn.Conv2d(num_channels, 6, kernel_size=5, padding=4, dilation=2)
        self.conv1c = nn.Conv2d(num_channels, 6, kernel_size=5, padding=6, dilation=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block (16 filters, 5x5 kernel, dilation 1, 2, 3)
        self.conv2a = nn.Conv2d(6, 16, kernel_size=5, padding=2, dilation=1)
        self.conv2b = nn.Conv2d(6, 16, kernel_size=5, padding=4, dilation=2)
        self.conv2c = nn.Conv2d(6, 16, kernel_size=5, padding=6, dilation=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Convolutional Block (26 filters, 5x5 kernel, dilation 1, 2, 3)
        self.conv3a = nn.Conv2d(16, 26, kernel_size=5, padding=2, dilation=1)
        self.conv3b = nn.Conv2d(16, 26, kernel_size=5, padding=4, dilation=2)
        self.conv3c = nn.Conv2d(16, 26, kernel_size=5, padding=6, dilation=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically compute the flattened size before defining fully connected layers
        self.flatten_size = self._get_flatten_size(num_channels, self.img_size)

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 700)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(700, 600)
        self.fc3 = nn.Linear(600, num_classes)

    def forward(self, x):
        # First Convolutional Block (dilation 1, 2, 3)
        x1 = F.relu(self.conv1a(x))
        x2 = F.relu(self.conv1b(x))
        x3 = F.relu(self.conv1c(x))
        x = torch.max(torch.max(x1, x2), x3)
        x = self.pool1(x)

        # Second Convolutional Block (dilation 1, 2, 3)
        x1 = F.relu(self.conv2a(x))
        x2 = F.relu(self.conv2b(x))
        x3 = F.relu(self.conv2c(x))
        x = torch.max(torch.max(x1, x2), x3)
        x = self.pool2(x)

        # Third Convolutional Block (dilation 1, 2, 3)
        x1 = F.relu(self.conv3a(x))
        x2 = F.relu(self.conv3b(x))
        x3 = F.relu(self.conv3c(x))
        x = torch.max(torch.max(x1, x2), x3)
        x = self.pool3(x)

        # Fully Connected Layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, x

    def _get_flatten_size(self, input_channels, image_size):
        """ Dynamically computes the flattened size after convolutional layers. """
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, image_size[0], image_size[1])
            x1 = F.relu(self.conv1a(dummy_input))
            x2 = F.relu(self.conv1b(dummy_input))
            x3 = F.relu(self.conv1c(dummy_input))
            x = torch.max(torch.max(x1, x2), x3)
            x = self.pool1(x)

            x1 = F.relu(self.conv2a(x))
            x2 = F.relu(self.conv2b(x))
            x3 = F.relu(self.conv2c(x))
            x = torch.max(torch.max(x1, x2), x3)
            x = self.pool2(x)

            x1 = F.relu(self.conv3a(x))
            x2 = F.relu(self.conv3b(x))
            x3 = F.relu(self.conv3c(x))
            x = torch.max(torch.max(x1, x2), x3)
            x = self.pool3(x)

            return x.numel()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Instantiate the model
    model = MSDCNN(num_channels=3, img_size=128, num_classes=2)
    
    # Print the model summary
    print(model)
    
    # Compute total number of parameters (same as original paper: 5,121,596)
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params}")