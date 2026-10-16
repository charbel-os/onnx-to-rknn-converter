import os
import torch
import torch.nn as nn
from rknn.api import RKNN

# ================= 1. Model Definition =================
class ImageRestorer150to640(nn.Module):
    """ Lightweight CNN Restorer + Upscaler (150x150 -> 640x640) """
    def __init__(self):
        super(ImageRestorer150to640, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(size=(640, 640), mode='bilinear', align_corners=False)
        
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() # Keeps output strictly bounded between 0.0 and 1.0

    def forward(self, x):
        # 1. Normalize input from [0, 255] to [0.0, 1.0]
        x = x / 255.0
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.relu(self.conv3(x))
        
        # 2. Output clean normalized range [0.0, 1.0] using Sigmoid
        x = self.sigmoid(self.conv4(x))
        return x

def main():
    onnx_path = "image_restorer_150to640.onnx"
    rknn_path = "image_restorer_150to640.rknn"

    # 2. Export PyTorch model to ONNX
    print("--> Exporting PyTorch model to ONNX...")
    model = ImageRestorer150to640().eval()
    dummy_input = torch.randn(1, 3, 150, 150)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Exported ONNX model to {onnx_path}")

    # 3. Convert ONNX to RKNN for RK3588
    print("--> Initializing RKNN Toolkit...")
    rknn = RKNN(verbose=True)

    # Configure RK3588 target platform
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
        target_platform='rk3588'
    )

    # Load ONNX graph
    print("--> Loading ONNX model into RKNN...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("Error loading ONNX model!")
        exit(1)

    # Build RKNN binary model
    print("--> Building RKNN model...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Error building RKNN model!")
        exit(1)

    # Export compiled model file
    print(f"--> Exporting RKNN binary to {rknn_path}...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("Error exporting RKNN binary!")
        exit(1)

    print(f"[SUCCESS] Successfully built and exported: {rknn_path}")
    rknn.release()

if __name__ == '__main__':
    main()
