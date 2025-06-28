import torch
import torch.nn as nn

class InceptionModule1D_DimReduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule1D_DimReduce, self).__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

        # Branch 2: 1x1 conv → 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Branch 3: 1x1 conv → 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # Branch 4: 3x3 maxpool → 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # (B, 4*out_channels, L)
