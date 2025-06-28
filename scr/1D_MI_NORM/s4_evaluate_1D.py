import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_1D import ECG1DDataset  # Import the ECG1DDataset class
# from model_1D import ECG1DCNN as MODEL # Import the SimpleECGCNN class
# from model_cnn_lstm import CNN1D_LSTM as MODEL
# from model_VGG1D import VGG1D as MODEL# Import the VGG1D model 
from model_ResNet18 import ResNet18_1D as MODEL  # Import the ResNet18 model 
# from QAT_ResNet18 import ResNet18_1D_QAT as MODEL, fuse_model
# from model_inception import Inception1D as MODEL

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil  # Đo CPU usage
import os
import torch.quantization as quant

# ==== Parameters ====
data_path = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_1D")
model_path = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet18_model.pth")
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==== Dataset và DataLoader ====
test_dataset = ECG1DDataset(root_dir=data_path/'test')  
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(f"Test set size: {len(test_dataset)}")

# ==== Load model ====
model = MODEL(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ==== Evaluation ====
all_preds = []
all_labels = []
all_probs = []  # Để lưu xác suất lớp MI (class 1)

total_infer_time = 0.0
process = psutil.Process(os.getpid())  # For CPU usage

with torch.no_grad():
    cpu_usage_start = process.cpu_percent(interval=None)  # Bắt đầu đo CPU

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()

        outputs = model(inputs)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        infer_time = end_time - start_time
        total_infer_time += infer_time

        probs = torch.softmax(outputs, dim=1)[:, 1]  # Xác suất cho class 1 (MI)
        preds = torch.argmax(outputs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    cpu_usage_end = process.cpu_percent(interval=None)  # Kết thúc đo CPU

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==== Metrics ====
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
auc = roc_auc_score(all_labels, all_probs)  # AUC sử dụng xác suất lớp MI
report = classification_report(all_labels, all_preds, target_names=['NORM', 'MI'])
cm = confusion_matrix(all_labels, all_preds)

print("===== Test Results =====")
print(f"Accuracy       : {acc:.4f}")
print(f"F1-score (macro): {f1:.4f}")
print(f"AUC-ROC       : {auc:.4f}")
print("\nClassification Report:\n", report)

# ==== Confusion Matrix Plot ====
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORM','MI'], yticklabels=['NORM','MI'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix_test.png')
plt.show()

# ==== ROC Curve Plot ====
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('roc_curve_test.png')
plt.show()

# ==== Performance Measurement ====
num_samples = len(test_dataset)
avg_latency = (total_infer_time / (num_samples / batch_size)) * 1000  # ms per batch
model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

print("===== Model & Performance Info =====")
print(f"Average Latency per batch : {avg_latency:.3f} ms")
print(f"Model Size               : {model_size:.2f} MB")

if device.type == 'cuda':
    gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
    gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)    # MB
    print(f"GPU Memory Allocated     : {gpu_mem_allocated:.2f} MB")
    print(f"GPU Memory Reserved      : {gpu_mem_reserved:.2f} MB")
else:
    print("GPU Memory Usage        : Not Applicable (CPU Mode)")

cpu_usage_avg = (cpu_usage_start + cpu_usage_end) / 2  # đơn giản lấy trung bình 2 lần đo
print(f"CPU Usage               : {cpu_usage_avg:.2f}%")


# ==== Inference Time Report ==== 
total_time_sec = total_infer_time
avg_time_per_sample = (total_time_sec / num_samples) * 1000  # ms

print("\n===== Inference Time Stats =====")
print(f"Total Inference Time      : {total_time_sec:.3f} sec")
print(f"Average Time per Sample   : {avg_time_per_sample:.3f} ms/sample")
