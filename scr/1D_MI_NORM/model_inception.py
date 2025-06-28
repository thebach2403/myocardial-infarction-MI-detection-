# import torch
# import torch.nn as nn

# class InceptionModule1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(InceptionModule1D, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
#             nn.ReLU()
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
#             nn.ReLU()
#         )
#         self.branch4 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
#             nn.Conv1d(in_channels, out_channels, kernel_size=1),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         b1 = self.branch1(x)
#         b2 = self.branch2(x)
#         b3 = self.branch3(x)
#         b4 = self.branch4(x)
#         return torch.cat([b1, b2, b3, b4], dim=1)  # (B, 4*out_channels, L)

# class Inception1D(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Inception1D, self).__init__()
#         self.stage1 = InceptionModule1D(in_channels=12, out_channels=16)   # → (B, 64, L)
#         self.stage2 = InceptionModule1D(in_channels=64, out_channels=32)   # → (B, 128, L)
#         self.stage3 = InceptionModule1D(in_channels=128, out_channels=32)  # → (B, 128, L)
        
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.global_pool(x)
#         x = self.classifier(x)
#         return x


import torch
import torch.nn as nn

# Module Inception cải tiến với BatchNorm và khả năng giảm chiều dài
class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(InceptionModule1D, self).__init__()
        
        # Nhánh 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # Nhánh 2: 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # Nhánh 3: 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # Nhánh 4: MaxPool + 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # Kết hợp 4 nhánh: (B, 4*out_channels, L)

# Mạng chính Inception1D cải tiến
class Inception1D(nn.Module):
    def __init__(self, num_classes=2):
        super(Inception1D, self).__init__()

        # Stage 1: đầu vào là 12 lead → tăng lên 128 kênh (32*4)
        self.stage1 = InceptionModule1D(in_channels=12, out_channels=32, stride=2)   # Output: (B, 128, 2048)

        # Stage 2: tăng thêm kênh và giảm độ dài tiếp
        self.stage2 = InceptionModule1D(in_channels=128, out_channels=64, stride=2)  # Output: (B, 256, 1024)

        # Stage 3: tiếp tục tăng đặc trưng
        self.stage3 = InceptionModule1D(in_channels=256, out_channels=128, stride=2)  # Output: (B, 512, 512)

        # Global average pooling để thu gọn L về 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output: (B, 512, 1)

        # Classifier cuối cùng
        self.classifier = nn.Sequential(
            nn.Flatten(),               # → (B, 512)
            nn.Linear(512, 128),        # Lớp ẩn
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) # Output logits
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
