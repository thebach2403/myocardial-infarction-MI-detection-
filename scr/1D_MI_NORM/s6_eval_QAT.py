import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_1D import ECG1DDataset
from QAT_ResNet18 import ResNet18_1D_QAT, fuse_model
import torch.quantization as quant
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.serialization import safe_globals
from torch.serialization import add_safe_globals
import torch.ao.nn.quantized as nnq  # quan trọng!
from torch.nn.quantized import Quantize, DeQuantize
import torch.serialization
from torch.ao.nn.intrinsic.quantized import ConvReLU1d
from torch.nn import Identity
from torch.nn.quantized import FloatFunctional
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.nn.intrinsic.quantized import ConvReLU1d
from QAT_ResNet18 import ResNet18_1D_QAT, BasicBlock1D
from torch.ao.nn.quantized import Quantize
from torch.nn import Identity, MaxPool1d
from torch.nn import Sequential  # Thêm import này
from torch.ao.nn.quantized.modules.conv import Conv1d as QConv1d
from torch.ao.nn.quantized.modules.functional_modules import QFunctional
from torch.nn import BatchNorm1d , AdaptiveAvgPool1d #
from torch.ao.nn.quantized.modules.linear import Linear as QLinear, LinearPackedParams
from torch._C import ScriptObject

# Hyperparameters
batch_size = 32
#  Phải ép về CPU khi load model quantized (vì convert xong chỉ chạy được CPU).
device = torch.device('cpu')  # QAT quantized model chỉ chạy trên CPU

# Load Test Data
test_dataset = ECG1DDataset(root_dir="E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_1D/test")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# torch.serialization.add_safe_globals([
#     ResNet18_1D_QAT,
#     BasicBlock1D,
#     QuantStub,
#     DeQuantStub,
#     FloatFunctional,
#     Identity,
#     ConvReLU1d,
#     Quantize,   # THÊM DÒNG NÀY
#     DeQuantize,  
#     MaxPool1d,   # ✅ THÊM CÁI NÀY ĐỂ HẾT LỖI HIỆN TẠI
#     Sequential,
#     QConv1d,
#     QFunctional,
#     BatchNorm1d,
#     AdaptiveAvgPool1d,
#     QLinear, 
#     LinearPackedParams,
#     ScriptObject
# ])

model = ResNet18_1D_QAT(num_classes=2)
model.eval()
fuse_model(model)
model.train()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)
model.eval()
quantized_model = quant.convert(model, inplace=False)  # Convert xong mới load state_dict int8

# Load int8 state_dict
checkpoint = torch.load("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet18_QAT.pth", map_location=device)
quantized_model.load_state_dict(checkpoint)
quantized_model.eval()

# Evaluation
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0
all_targets = []
all_preds = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = quantized_model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_targets.extend(target.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")

# Optional: Confusion Matrix + Classification Report
cm = confusion_matrix(all_targets, all_preds)
report = classification_report(all_targets, all_preds, digits=4)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)


import time

# === Inference time đo trên toàn bộ test_loader ===
start_time = time.time()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        _ = quantized_model(data)
end_time = time.time()

avg_inference_time = (end_time - start_time) / len(test_loader)
print(f"\n⏱️ Average inference time per batch (batch_size={batch_size}): {avg_inference_time:.4f} seconds")

# === Thống kê số lượng tham số ===
total_params = sum(p.numel() for p in quantized_model.parameters())
trainable_params = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)

print(f"\n📦 Total parameters: {total_params:,}")
print(f"🛠️  Trainable parameters: {trainable_params:,}")
