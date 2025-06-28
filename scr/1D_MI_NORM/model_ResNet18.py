import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    expansion = 1  # for BasicBlock, output_channels = planes * expansion

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_1D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18_1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)  # [B, 64, 2048]
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)                # [B, 64, 1024]

        self.layer1 = self._make_layer(64, 2, stride=1)    # giữ nguyên [B,64,1024]
        self.layer2 = self._make_layer(128, 2, stride=2)   # giảm còn [B,128,512]
        self.layer3 = self._make_layer(256, 2, stride=2)   # [B,256,256]
        self.layer4 = self._make_layer(512, 2, stride=2)   # [B,512,128]

        self.avgpool = nn.AdaptiveAvgPool1d(1)             # [B,512,1]
        self.fc = nn.Linear(512, num_classes)              # [B,2]

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # [B, 64, 2048]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, 1024]

        x = self.layer1(x)  # [B, 64, 1024]
        x = self.layer2(x)  # [B, 128, 512]
        x = self.layer3(x)  # [B, 256, 256]
        x = self.layer4(x)  # [B, 512, 128]

        x = self.avgpool(x) # [B, 512, 1]
        x = torch.flatten(x, 1) # [B, 512]
        x = self.fc(x)      # [B, 2]

        return x
