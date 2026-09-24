import sys
import os
import glob
import json
import cv2
import numpy as np
import onnx
from rknn.api import RKNN
from ultralytics import YOLO

def main():
    img_size = 640
    rknn_output_path = f"yolov8s_{img_size}_hybrid.rknn"
    calib_dataset_path = "dataset640.txt"

    print(f"--> Step 1: Loading YOLOv8s pre-trained model (imgsz={img_size})...")
    model = YOLO("yolov8s.pt")

    print(f"--> Step 2: Exporting model to ONNX (imgsz={img_size})...")
    onnx_path = model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"--> ONNX model generated successfully at: {onnx_path}")

    print("--> Step 3: Configuring NPU target & pixel normalization for RK3588...")
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

    # Verify Calibration Dataset Exists
    if not os.path.exists(calib_dataset_path):
        print(f"❌ ERROR: Calibration file '{calib_dataset_path}' not found!")
        sys.exit(1)

    print("--> Step 5: Running Hybrid Quantization Step 1 (Generating config template)...")
    ret = rknn.build(do_quantization=True, dataset=calib_dataset_path, hybrid_quantization_step=1)
    if ret != 0:
        print("❌ ERROR: Hybrid quantization step 1 failed.")
        sys.exit(1)

    print("--> Step 6: Automatically configuring output layers to FP16 in quantization config...")
    cfg_files = glob.glob("*.quantization.cfg")
    if not cfg_files:
        print("❌ ERROR: No .quantization.cfg file found after step 1!")
        sys.exit(1)
    
    cfg_file = cfg_files[0]
    print(f"--> Found config file: {cfg_file}")

    # Inspect ONNX outputs to dynamically target final detection head layers
    onnx_model = onnx.load(onnx_path)
    output_names = [output.name for output in onnx_model.graph.output]
    print(f"--> Detected ONNX output nodes: {output_names}")

    # Map final output layers to float16 to preserve precision
    custom_layers_dict = {name: "float16" for name in output_names}
    replacement_str = "custom_quantize_layers: " + json.dumps(custom_layers_dict)

    with open(cfg_file, "r") as f:
        cfg_content = f.read()

    if "custom_quantize_layers: {}" in cfg_content:
        cfg_content = cfg_content.replace("custom_quantize_layers: {}", replacement_str)
    else:
        cfg_content += f"\n{replacement_str}\n"

    with open(cfg_file, "w") as f:
        f.write(cfg_content)
    print(f"--> Successfully injected FP16 overrides for: {custom_layers_dict}")

    print("--> Step 7: Running Hybrid Quantization Step 2 (Building hybrid model)...")
    ret = rknn.build(do_quantization=True, dataset=calib_dataset_path, hybrid_quantization_step=2)
    if ret != 0:
        print("❌ ERROR: Hybrid quantization step 2 failed.")
        sys.exit(1)

    print(f"--> Step 8: Exporting Hybrid RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("❌ ERROR: Model export failed.")
        sys.exit(1)

    print(f"Success! Hybrid YOLOv8s RKNN model built and saved to {rknn_output_path}.")
    rknn.release()

if __name__ == "__main__":
    main()
