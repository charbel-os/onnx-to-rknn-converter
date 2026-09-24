import sys
from rknn.api import RKNN
from ultralytics import YOLO

def main():
    img_size = 416
    # Change output name to reflect INT8 quantization
    rknn_output_path = f"yolov8s_{img_size}_int8.rknn"
    
    # Path to your calibration text file (relative or absolute)
    # Ensure your dataset.txt lists paths to images inside your calib_img directory
    calib_dataset_path = "dataset.txt"

    print(f"--> Step 1: Loading YOLOv8s pre-trained model (imgsz={img_size})...")
    model = YOLO("yolov8s.pt")

    print(f"--> Step 2: Exporting model to ONNX (imgsz={img_size})...")
    onnx_path = model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"--> ONNX model generated successfully at: {onnx_path}")

    print("--> Step 3: Configuring NPU target & pixel normalization for ROCK 5C (RK3588)...")
    rknn = RKNN(verbose=True)
    
    rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]]
    )

    print(f"--> Step 4: Loading ONNX model from: {onnx_path}")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("Failed to load ONNX model.")
        sys.exit(1)

    print("--> Step 5: Building INT8 RKNN model with calibration dataset...")
    # CRITICAL: Enable quantization and pass the dataset.txt file for INT8 calibration
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
