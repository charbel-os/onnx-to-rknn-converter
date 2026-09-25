import sys
import os
from rknn.api import RKNN

def main():
    img_size = 416
    onnx_path = "yolov7-tiny.onnx"
    rknn_output_path = f"yolov7-tiny_{img_size}_fp16.rknn"

    print(f"--> Step 1: Checking ONNX model at: {onnx_path}")
    if not os.path.exists(onnx_path):
        print(f"❌ ERROR: ONNX model '{onnx_path}' not found.")
        sys.exit(1)

    print("--> Step 2: Configuring NPU target & pixel normalization for RK3588 (FP16)...")
    rknn = RKNN(verbose=True)
    
    rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        optimization_level=3
    )

    print(f"--> Step 3: Loading ONNX model into RKNN...")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("❌ ERROR: Failed to load ONNX model.")
        sys.exit(1)

    print("--> Step 4: Building RKNN model in FP16 precision (No Quantization/Hybrids)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("❌ ERROR: FP16 model build failed.")
        sys.exit(1)

    print(f"--> Step 5: Exporting FP16 RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("❌ ERROR: Model export failed.")
        sys.exit(1)

    print(f"Success! FP16 YOLOv7-tiny RKNN model saved as '{rknn_output_path}'.")
    rknn.release()

if __name__ == "__main__":
    main()
