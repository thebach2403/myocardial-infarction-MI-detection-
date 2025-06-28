import torch
import torch.quantization as quant
from QAT_ResNet18 import ResNet18_1D_QAT, fuse_model

# 1. Load model kiến trúc
model = ResNet18_1D_QAT(num_classes=2)
model.eval()

# 2. Fuse và QAT chuẩn bị (giống lúc train)
fuse_model(model)
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model.train()
quant.prepare_qat(model, inplace=True)
model.eval()

# 3. Convert to quantized model
quantized_model = quant.convert(model, inplace=False)

# 4. Load state_dict INT8 đã train xong (đã QAT)
checkpoint = torch.load("E:/pv/WORKING/ECG_main_folder/ECG_Classification_MI_detect/models/ResNet18_QAT.pth", map_location='cpu')
quantized_model.load_state_dict(checkpoint)
quantized_model.eval()

# 5. Export to ONNX
dummy_input = torch.randn(1, 12, 4096)  # batch=1, 12 lead, 4096 sample (chỉnh theo input của bạn)

torch.onnx.export(
    quantized_model,             # model đã QAT và convert
    dummy_input,                 # input giả lập
    "ResNet18_QAT.onnx",         # file output
    export_params=True,
    opset_version=13,            # hoặc 17 nếu muốn (tuỳ Jetson hoặc inference tool)
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✅ Convert to ONNX hoàn tất: ResNet18_QAT.onnx")
