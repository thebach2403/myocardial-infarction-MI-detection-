import torch
from model_resnet1D import ResNet1D

# ==== Load trained model ====
model = ResNet1D(num_classes=2)
model.load_state_dict(torch.load("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet1D_model.pth"))
model.eval()

# ==== Dummy Input ====
dummy_input = torch.randn(1, 12, 4096)  # (batch_size, channels, samples) – chỉnh lại nếu khác

# ==== Export to ONNX ====
onnx_path = "E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet1D_model.onnx"

torch.onnx.export(
    model,                      # model cần convert
    dummy_input,                # dummy input
    onnx_path,                  # lưu output file
    export_params=True,         # xuất trọng số
    opset_version=11,           # opset version (11/12/13 đều ok cho Jetson Nano)
    do_constant_folding=True,   # tối ưu hằng số
    input_names=['input'],      # tên input
    output_names=['output'],    # tên output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # batch size dynamic
)

print(f"✅ Model converted to ONNX and saved at {onnx_path}")
