import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D_LSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN1D_LSTM, self).__init__()
        
        # 1D CNN để trích đặc trưng trên mỗi lead
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, stride=1, padding=3)  # giữ nguyên chiều dài
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # giảm 1/2 chiều dài -> 4096 -> 2048

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 2048 -> 1024

        # LSTM input size = số channels output từ CNN (64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # Fully connected
        self.fc1 = nn.Linear(128*2, 64)  # 128 * 2 vì bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 12, 4096]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # [batch_size, 32, 2048]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # [batch_size, 64, 1024]

        # LSTM expects input [batch, seq_len, feature]
        x = x.permute(0, 2, 1)  # [batch_size, 1024, 64]

        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, 1024, 256]
        
        # Lấy hidden state cuối cùng của 2 chiều (forward + backward)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # [batch_size, 256]

        out = self.fc1(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc2(out)  # [batch_size, num_classes]
        
        return out
