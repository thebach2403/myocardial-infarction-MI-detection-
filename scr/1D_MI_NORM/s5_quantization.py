import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_1D import ECG1DDataset
from QAT_ResNet18 import ResNet18_1D_QAT as MODEL, fuse_model
import torch.quantization as quant
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
num_epochs_qat = 5
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
train_dataset = ECG1DDataset(root_dir="E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_1D/train")
val_dataset = ECG1DDataset(root_dir="E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/data/preprocessed_MI_1D/val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model
model = MODEL(num_classes=2).to(device)
model.eval()  # Fuse ở chế độ eval
fuse_model(model)
model.train()  # Về train để QAT

# Warm-up Float Training
float_train_epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(float_train_epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
print("Float training warm-up done. Now start QAT.")

# QAT Preparation
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_loss = float('inf')
best_state_dict = None

# QAT Training Loop
for epoch in range(num_epochs_qat):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()
        total_train += target.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_running_loss, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for val_data, val_target in val_loader:
            val_data, val_target = val_data.to(device), val_target.to(device)
            val_output = model(val_data)
            val_loss = criterion(val_output, val_target)
            val_running_loss += val_loss.item()
            _, val_predicted = torch.max(val_output.data, 1)
            correct_val += (val_predicted == val_target).sum().item()
            total_val += val_target.size(0)

    avg_val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs_qat}] - "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step(avg_val_loss)
    print(f"Learning rate = {optimizer.param_groups[0]['lr']}")

    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state_dict = model.state_dict().copy()

# Convert to quantized model

# Load lại best weight vào CPU model
model.load_state_dict(best_state_dict)
# Chuyển sang chế độ đánh giá (eval) và về CPU
model.eval()
model.cpu()
# Convert (QAT → quantized model)
quantized_model = quant.convert(model, inplace=False)

# Save statedict của qat model
torch.save(quantized_model.state_dict(), "E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet18_QAT.pth")
# Save full object (chỉ để dùng cho test nội bộ)
torch.save(quantized_model, "E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet18_QAT_FULL.pth")
print("Best Quantized model saved as ResNet18_QAT.pth")

# Plot Loss Curve
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs_qat+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs_qat+1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('QAT Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/figures/QAT_loss_curve.png')
plt.show()