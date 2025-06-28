import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
random.seed(44)
np.random.seed(44)
torch.manual_seed(4)
torch.cuda.manual_seed(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from pathlib import Path 
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score

from dataset_1D import ECG1DDataset  # Import the ECG1DDataset class
# from model_1D import ECG1DCNN  # Import the SimpleECGCNN class
# from model_cnn_lstm import CNN1D_LSTM
# from model_resnet1D import ResNet1D as MODEL  # Import the ResNet1D model 
from model_VGG1D import VGG1D as MODEL  # Import the VGG1D model 
# from model_ResNet18 import ResNet18_1D as MODEL  # Import the ResNet18 model 
# from model_inception import Inception1D as MODEL

torch.cuda.empty_cache()# Giải phóng bộ nhớ GPU trước khi bắt đầu huấn luyện 