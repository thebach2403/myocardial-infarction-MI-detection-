import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_1D import ECG1DDataset  # Import the ECG1DDataset class
# from model_1D import ECG1DCNN  # Import the SimpleECGCNN class
# from model_cnn_lstm import CNN1D_LSTM
from model_resnet1D import ResNet1D  # Import the ResNet1D class 

from tqdm import tqdm
from pathlib import Path 
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score

torch.cuda.empty_cache()# Giải phóng bộ nhớ GPU trước khi bắt đầu huấn luyện 

import random
random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
torch.cuda.manual_seed(44)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###################### Parameters #########################
# ===================== Training Config =====================
data_path = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_subclass")
save_path = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/CNN1D_model.pth")

batch_size = 32
num_epochs = 40
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== Data Loaders =====================
train_dataset = ECG1DDataset(root_dir=data_path/'train')
val_dataset = ECG1DDataset(root_dir=data_path/'val')
test_dataset = ECG1DDataset(root_dir=data_path/'test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples  : {len(val_dataset)}")
print(f"Test samples : {len(test_dataset)}")

# ===================== Model, Loss, Optimizer =====================
model = ResNet1D(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=5)

# ===================== Evaluation =====================
def evaluate(dataloader):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss / len(dataloader), f1

# ===================== Training Loop =====================
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # chèn clip gradient để tránh exploding gradients cho LSTM 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    val_loss, val_f1 = evaluate(val_loader)
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss)

    print(f"📊 Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}, Train Acc={acc:.2f}%")
    scheduler.step(val_loss)
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("⛔ Early stopping.")
        break

# ===================== Save Model & Plot =====================
torch.save(model.state_dict(), save_path)
print("✅ Model saved at:", save_path)

plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.title('Training/Validation Loss')
plt.savefig('CNN1D_loss_curve.png')
plt.show()
