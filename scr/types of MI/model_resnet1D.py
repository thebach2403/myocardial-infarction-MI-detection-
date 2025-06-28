import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2  # giữ nguyên kích thước

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample  # dùng khi in_channels != out_channels hoặc stride > 1

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Nếu cần downsample
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3)  # [B, 64, 2048]
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)     # [B, 64, 1024]

        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)  # [B, 128, 512]
        self.layer3 = self._make_layer(128, 256, stride=2) # [B, 256, 256]

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # [B, 256, 1]
        self.fc = nn.Linear(256, num_classes)   # [B, 2]

    def _make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        return ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample)

    def forward(self, x):
        x = self.conv1(x)   # [B, 64, 2048]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, 1024]

        x = self.layer1(x)  # [B, 64, 1024]
        x = self.layer2(x)  # [B, 128, 512]
        x = self.layer3(x)  # [B, 256, 256]

        x = self.avgpool(x) # [B, 256, 1]
        x = x.squeeze(-1)   # [B, 256]

        x = self.fc(x)      # [B, 2]
        return x
