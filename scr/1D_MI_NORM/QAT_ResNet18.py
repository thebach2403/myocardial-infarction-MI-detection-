# import torch.quantization
# import torch
# import torch.nn as nn
# import torch.nn.quantized as nnq

# class BasicBlock1D(nn.Module):
#     expansion = 1  # for BasicBlock, output_channels = planes * expansion

#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super(BasicBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample

#         self.add = nnq.FloatFunctional()

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = self.add.add(out, identity)  # CHỖ QUAN TRỌNG
#         out = self.relu(out)

#         return out

# class ResNet18_1D_QAT(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ResNet18_1D_QAT, self).__init__()
#         self.in_planes = 64

#         self.quant = torch.quantization.QuantStub()   # thêm QuantStub
#         self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(64, 2, stride=1)
#         self.layer2 = self._make_layer(128, 2, stride=2)
#         self.layer3 = self._make_layer(256, 2, stride=2)
#         self.layer4 = self._make_layer(512, 2, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(512, num_classes)
#         self.dequant = torch.quantization.DeQuantStub()  # thêm DeQuantStub

#     def _make_layer(self, planes, blocks, stride):
#         downsample = None
#         if stride != 1 or self.in_planes != planes:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes),
#             )

#         layers = []
#         layers.append(BasicBlock1D(self.in_planes, planes, stride, downsample))
#         self.in_planes = planes
#         for _ in range(1, blocks):
#             layers.append(BasicBlock1D(planes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.quant(x)   # quantize input
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.dequant(x) # dequantize output
#         return x

# def fuse_model(model):
#     # Fuse conv+bn+relu trong conv1
#     torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
#     # Fuse cho từng BasicBlock
#     for module_name, module in model.named_children():
#         if "layer" in module_name:
#             for basic_block in module:
#                 torch.quantization.fuse_modules(basic_block,
#                     [['conv1', 'bn1', 'relu'],
#                      ['conv2', 'bn2']], inplace=True)
#     return model

import torch.quantization
import torch
import torch.nn as nn
import torch.nn.quantized as nnq

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

        self.add = nnq.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add.add(out, identity)  # quan trọng cho QAT
        out = self.relu(out)

        return out


class ResNet18_1D_QAT(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18_1D_QAT, self).__init__()
        self.in_planes = 64

        self.quant = torch.quantization.QuantStub()   # thêm QuantStub
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = torch.quantization.DeQuantStub()  # thêm DeQuantStub

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
        x = self.quant(x)  # Quantize input

        # Sau khi fuse, conv1 + bn1 + relu đã được gộp thành 1 module (ConvReLU1d)
        # nên ta chỉ gọi 1 lần conv1 thôi, không cần gọi bn1 và relu riêng nữa
        x = self.conv1(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)  # Dequantize output
        return x


def fuse_model(self):
    for m in self.modules():
        if type(m) == BasicBlock1D:
            torch.ao.quantization.fuse_modules(m, ['conv1', 'bn1', 'relu'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv2', 'bn2'], inplace=True)
            if m.downsample is not None:
                # Quan trọng: fuse downsample Conv+BN ở shortcut path
                torch.ao.quantization.fuse_modules(m.downsample, ['0', '1'], inplace=True)
