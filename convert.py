import sys
import os
import glob
import json
import shutil
import onnx
from rknn.api import RKNN
from ultralytics import YOLO

def main():
    img_size = 640
    rknn_output_path = f"yolov8s_{img_size}_hybrid.rknn"
    calib_dataset_path = "dataset640.txt"
    debug_artifact_dir = "rknn_debug_artifacts"

    # Create directory to collect all files needed for troubleshooting
    os.makedirs(debug_artifact_dir, exist_ok=True)

    repo_root = os.getcwd()
    full_calib_path = os.path.join(repo_root, calib_dataset_path)

    print(f"--> Repository Root: {repo_root}")
    print(f"--> Calibration Dataset File: {full_calib_path}")

    print(f"--> Step 1: Loading YOLOv8s pre-trained model (imgsz={img_size})...")
    model = YOLO("yolov8s.pt")

    print(f"--> Step 2: Exporting model to ONNX (imgsz={img_size})...")
    onnx_path = model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"--> ONNX model generated successfully at: {onnx_path}")

    # Copy ONNX file to debug artifacts
    if os.path.exists(onnx_path):
        shutil.copy(onnx_path, os.path.join(debug_artifact_dir, os.path.basename(onnx_path)))

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

    if not os.path.exists(full_calib_path):
        print(f"❌ ERROR: Calibration file '{full_calib_path}' not found at repository root!")
        sys.exit(1)

    print("--> Step 5: Running Hybrid Quantization Step 1 (Generating config template)...")
    ret = rknn.hybrid_quantization_step1(dataset=full_calib_path, proposal=True)
    if ret != 0:
        print("❌ ERROR: Hybrid quantization step 1 failed.")
        sys.exit(1)

    print("--> Step 6: Collecting intermediate configuration & model files...")
    cfg_files = glob.glob("*.quantization.cfg")
    model_files = glob.glob("*.model")
    data_files = glob.glob("*.data")

    if not cfg_files or not model_files or not data_files:
        print("❌ ERROR: Required intermediate files missing after Step 1!")
        sys.exit(1)
    
    cfg_file = cfg_files[0]
    model_file = model_files[0]
    data_file = data_files[0]

    # Copy intermediate files into the debug folder before modification
    shutil.copy(cfg_file, os.path.join(debug_artifact_dir, "initial_" + cfg_file))
    shutil.copy(model_file, os.path.join(debug_artifact_dir, model_file))
    shutil.copy(data_file, os.path.join(debug_artifact_dir, data_file))

    # Inspect ONNX outputs to dynamically target final detection head layers
    onnx_model = onnx.load(onnx_path)
    output_names = [output.name for output in onnx_model.graph.output]
    print(f"--> Detected ONNX output nodes: {output_names}")

    custom_layers_dict = {name: "float16" for name in output_names}

    # Line-by-line safe injection for float16 override configuration
    with open(cfg_file, "r") as f:
        lines = f.readlines()

    updated_lines = []
    found_custom_layer_prop = False
    for line in lines:
        if line.strip().startswith("custom_quantize_layers"):
            updated_lines.append(f"custom_quantize_layers: {json.dumps(custom_layers_dict)}\n")
            found_custom_layer_prop = True
        else:
            updated_lines.append(line)

    if not found_custom_layer_prop:
        updated_lines.append(f"\ncustom_quantize_layers: {json.dumps(custom_layers_dict)}\n")

    with open(cfg_file, "w") as f:
        f.writelines(updated_lines)

    # Copy the final modified configuration file to debug artifacts
    shutil.copy(cfg_file, os.path.join(debug_artifact_dir, "modified_" + cfg_file))
    print(f"--> Successfully injected FP16 overrides into config for: {custom_layers_dict}")

    print("--> Step 7: Running Hybrid Quantization Step 2 (Building hybrid model)...")
    ret = rknn.hybrid_quantization_step2(
        model_input=model_file,
        data_input=data_file,
        model_quantization_cfg=cfg_file
    )
    if ret != 0:
        print("❌ ERROR: Hybrid quantization step 2 failed.")
        sys.exit(1)

    print(f"--> Step 8: Exporting Hybrid RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("❌ ERROR: Model export failed.")
        sys.exit(1)

    # Copy final model output into debug collection folder
    shutil.copy(rknn_output_path, os.path.join(debug_artifact_dir, rknn_output_path))

    print(f"Success! Hybrid YOLOv8s RKNN model and all debug payloads prepared in '{debug_artifact_dir}/'.")
    rknn.release()

if __name__ == "__main__":
    main()
