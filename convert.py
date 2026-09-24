import sys
import os
import glob
import json
import onnx
from rknn.api import RKNN
from ultralytics import YOLO

def main():
    img_size = 640
    rknn_output_path = f"yolov8s_{img_size}_hybrid.rknn"
    dataset_dir = "dataset640"
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

    # Verify dataset directory exists in your repository
    if not os.path.exists(dataset_dir):
        print(f"❌ ERROR: Dataset directory '{dataset_dir}' not found in repository!")
        sys.exit(1)

    # Dynamically scan dataset640 folder to build absolute path text file for RKNN toolkit
    print(f"--> Step 5: Scanning '{dataset_dir}' directory for calibration images...")
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                abs_path = os.path.abspath(os.path.join(root, file))
                image_files.append(abs_path)

    if not image_files:
        print(f"❌ ERROR: No valid images found inside '{dataset_dir}' folder!")
        sys.exit(1)

    with open(calib_dataset_path, "w") as f:
        for img_path in image_files:
            f.write(img_path + "\n")
    
    print(f"--> Generated '{calib_dataset_path}' successfully with {len(image_files)} calibration images.")

    print("--> Step 6: Running Hybrid Quantization Step 1 (Generating config template)...")
    ret = rknn.hybrid_quantization_step1(dataset=calib_dataset_path, proposal=True)
    if ret != 0:
        print("❌ ERROR: Hybrid quantization step 1 failed.")
        sys.exit(1)

    print("--> Step 7: Automatically configuring output layers to FP16 in quantization config...")
    cfg_files = glob.glob("*.quantization.cfg")
    model_files = glob.glob("*.model")
    data_files = glob.glob("*.data")

    if not cfg_files or not model_files or not data_files:
        print("❌ ERROR: Required intermediate files (.quantization.cfg, .model, or .data) missing after Step 1!")
        sys.exit(1)
    
    cfg_file = cfg_files[0]
    model_file = model_files[0]
    data_file = data_files[0]
    
    print(f"--> Found config file: {cfg_file}")
    print(f"--> Found intermediate model file: {model_file}")

    # Inspect ONNX outputs to dynamically target final detection head layers
    onnx_model = onnx.load(onnx_path)
    output_names = [output.name for output in onnx_model.graph.output]
    print(f"--> Detected ONNX output nodes: {output_names}")

    # Map final output layers to float16 to preserve bounding-box precision
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

    print("--> Step 8: Running Hybrid Quantization Step 2 (Building hybrid model)...")
    ret = rknn.hybrid_quantization_step2(
        model_input=model_file,
        data_input=data_file,
        model_quantization_cfg=cfg_file
    )
    if ret != 0:
        print("❌ ERROR: Hybrid quantization step 2 failed.")
        sys.exit(1)

    print(f"--> Step 9: Exporting Hybrid RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("❌ ERROR: Model export failed.")
        sys.exit(1)

    print(f"Success! Hybrid YOLOv8s RKNN model built and saved to {rknn_output_path}.")
    rknn.release()

if __name__ == "__main__":
    main()
