import sys
import os
import torch
import torchvision
import onnx
from rknn.api import RKNN

# Add the cloned yolov7 directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov7')))
from models.experimental import attempt_load

def main():
    img_size = 416
    rknn_output_path = f"yolov7-tiny_{img_size}_fp16.rknn"
    onnx_path = "yolov7-tiny.onnx"

    print(f"--> Step 1: Loading YOLOv7-tiny pre-trained model using attempt_load (imgsz={img_size})...")
    model = attempt_load('yolov7-tiny.pt', map_location='cpu')
    model.eval()

    print(f"--> Step 2: Exporting PyTorch model to ONNX...")
    dummy_input = torch.randn(1, 3, img_size, img_size, device='cpu')
    
    # Fix for newer torch versions during ONNX export
    for k, m in model.named_modules():
        if type(m) is torch.nn.Upsample:
            m.recompute_scale_factor = None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output']
    )
    print(f"--> ONNX model generated successfully at: {onnx_path}")

    print("--> Step 3: Configuring NPU target & pixel normalization for RK3588 (FP16)...")
    rknn = RKNN(verbose=True)
    
    rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        optimization_level=3
    )

    print(f"--> Step 4: Loading ONNX model from: {onnx_path}")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("❌ ERROR: Failed to load ONNX model.")
        sys.exit(1)

    print("--> Step 5: Building RKNN model in FP16 precision (No Quantization/Hybrids)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("❌ ERROR: FP16 model build failed.")
        sys.exit(1)

    print(f"--> Step 6: Exporting FP16 RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("❌ ERROR: Model export failed.")
        sys.exit(1)

    print(f"Success! FP16 YOLOv7-tiny RKNN model saved as '{rknn_output_path}'.")
    rknn.release()

if __name__ == "__main__":
    main()
