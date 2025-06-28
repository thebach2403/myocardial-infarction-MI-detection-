import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock1D, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # Giảm 1/2 chiều dài
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VGG1D(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG1D, self).__init__()

        self.features = nn.Sequential(
            VGGBlock1D(12, 64, num_convs=2),    # Block 1: Conv 64 x2
            VGGBlock1D(64, 128, num_convs=2),   # Block 2: Conv 128 x2
            VGGBlock1D(128, 256, num_convs=3),  # Block 3: Conv 256 x3
            VGGBlock1D(256, 512, num_convs=3),  # Block 4: Conv 512 x3
            VGGBlock1D(512, 512, num_convs=3),  # Block 5: Conv 512 x3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * (4096 // (2**5)), 4096),  # sau 5 pool: 4096 / 32 = 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # 2 classes: MI / NORM
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
