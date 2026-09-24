import sys
import os
from rknn.api import RKNN
from ultralytics import YOLO

def main():
    img_size = 640
    rknn_output_path = f"yolov8s_{img_size}_int8.rknn"
    
    # Path to your calibration text file
    calib_dataset_path = "dataset640.txt"

    print(f"--> Step 1: Loading YOLOv8s pre-trained model (imgsz={img_size})...")
    model = YOLO("yolov8s.pt")

    print(f"--> Step 2: Exporting model to ONNX (imgsz={img_size})...")
    onnx_path = model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"--> ONNX model generated successfully at: {onnx_path}")

    print("--> Step 3: Configuring top-tier NPU target & pixel normalization for RK3588...")
    rknn = RKNN(verbose=True)
    
    rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        optimization_level=3
    )

    print(f"--> Step 4: Loading ONNX model from: {onnx_path}")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("Failed to load ONNX model.")
        sys.exit(1)

    # --- DEBUGGING BLOCK: Verify Dataset & Calibration Paths ---
    print(f"--> Debug: Checking calibration dataset: {calib_dataset_path}")
    if not os.path.exists(calib_dataset_path):
        print(f"❌ ERROR: Calibration file '{calib_dataset_path}' not found in working directory!")
        sys.exit(1)
        
    with open(calib_dataset_path, 'r') as f:
        calib_lines = [line.strip() for line in f if line.strip()]
        
    print(f"--> Debug: Successfully read {len(calib_lines)} lines from {calib_dataset_path}")
    if len(calib_lines) > 0:
        print(f"--> Debug: First image path listed: '{calib_lines[0]}'")
        if os.path.exists(calib_lines[0]):
            print(f"--> Debug: ✔ Verified first image exists on disk.")
        else:
            print(f"--> Warning: ⚠ First image path does NOT exist on disk! Check your relative paths.")
    else:
        print(f"❌ ERROR: Calibration file '{calib_dataset_path}' is empty!")
        sys.exit(1)
    # -----------------------------------------------------------

    print("--> Step 5: Building INT8 RKNN model with calibration dataset...")
    # When do_quantization=True and dataset is passed, RKNN-Toolkit2 reads this file line-by-line
    if rknn.build(do_quantization=True, dataset=calib_dataset_path) != 0:
        print("Build and quantization failed.")
        sys.exit(1)

    print(f"--> Step 6: Exporting INT8 RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("Export failed.")
        sys.exit(1)

    print(f"Success! YOLOv8s (imgsz={img_size}) INT8 RKNN model built successfully and saved to {rknn_output_path}.")
    rknn.release()

if __name__ == "__main__":
    main()
