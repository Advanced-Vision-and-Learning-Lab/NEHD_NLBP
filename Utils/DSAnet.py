import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding='same', bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv1(attention)
        attention = torch.sigmoid(attention)
        return x * attention

class NMNet(nn.Module):
    def __init__(self, num_channels, img_size, num_classes):
        super(NMNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(1, 10, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(10)
        self.spatial_attn1 = SpatialAttention(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(10, 12, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(12)
        self.spatial_attn2 = SpatialAttention(12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12 * ((img_size // 4) ** 2), 420)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(420, 250)
        self.fc3 = nn.Linear(250, num_classes)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.bn1(x)
        x = self.spatial_attn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = self.spatial_attn2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example usage:
# model = NMNet(num_channels=3, img_size=32, num_classes=10)

