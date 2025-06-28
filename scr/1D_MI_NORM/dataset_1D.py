
################################################################################################333

from torch.utils.data import Dataset
import os
import numpy as np
import torch

class ECG1DDataset(Dataset):  # quản lý raw 1D ECG signals
    def __init__(self, root_dir, transform=None):
    
        self.root_dir = root_dir
        self.transform = transform
        self.data_paths = []
        self.labels = []

        label_map = {'NORM': 0, 'MI': 1}

        # Duyệt qua các thư mục MI và NORM
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir): # kiểm tra xem label_dir có phải là một thư mục hay không 
                for filename in os.listdir(label_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(label_dir, filename)
                        self.data_paths.append(file_path)
                        self.labels.append(label_map[label])  # 0: NORM, 1: MI

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        signal = np.load(file_path)  # (12, 4096)

        # Chuyển về torch.Tensor
        signal = torch.tensor(signal, dtype=torch.float32)

        # Optional transform (nếu cần)
        if self.transform:
            signal = self.transform(signal)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)  # để dùng cho CrossEntropyLoss

        return signal, label

###################################################