import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_1D import ECG1DDataset  # Import the ECG1DDataset class
# from model_1D import ECG1DCNN  # Import the SimpleECGCNN class
# from model_cnn_lstm import CNN1D_LSTM 
from model_resnet1D import ResNet1D  # Import the ResNet1D class

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Parameters ====
data_path = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_subclass")
model_path = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/CNN1D_model.pth")
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==== Dataset và DataLoader ====
test_dataset = ECG1DDataset(root_dir=data_path/'test')  
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(f"Test set size: {len(test_dataset)}")

# ==== Load model ====
model = ResNet1D(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== Evaluation ====
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==== Metrics ====
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
report = classification_report(all_labels, all_preds, target_names=['NORM', 'MI'])
cm = confusion_matrix(all_labels, all_preds)

print("===== Test Results =====")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score (macro): {f1:.4f}")
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