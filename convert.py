import sys
import os
import cv2
import numpy as np
from rknn.api import RKNN

def create_dummy_dataset(dataset_path="dataset.txt", img_path="calib_img.jpg"):
    """Creates a dummy calibration image and dataset file if they don't exist."""
    if not os.path.exists(img_path):
        # Create a dummy 640x640 gray image for basic compilation
        dummy_img = np.full((640, 640, 3), 114, dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
    
    if not os.path.exists(dataset_path):
        with open(dataset_path, 'w') as f:
            f.write(f"{img_path}\n")

def convert_model(onnx_path, output_path, platform="rk3588", dataset_path="dataset.txt"):
    onnx_path = os.path.abspath(onnx_path)
    output_path = os.path.abspath(output_path)
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(onnx_path):
        print(f"ERROR: Could not find ONNX model at: {onnx_path}")
        sys.exit(1)

    # Ensure a calibration dataset file exists for INT8 quantization
    if not os.path.exists(dataset_path):
        print(f"--> Calibration dataset not found at {dataset_path}. Creating a dummy configuration...")
        create_dummy_dataset(dataset_path)

    rknn = RKNN(verbose=True)

    print(f'--> Configuring model settings for platform: {platform}')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform=platform
    )

    print(f'--> Loading ONNX model from: {onnx_path}')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Failed to load ONNX model!')
        sys.exit(1)

    print(f'--> Building model as INT8 (do_quantization=True) using dataset: {dataset_path}...')
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print('Failed to build RKNN INT8 model!')
        sys.exit(1)

    print(f'--> Exporting RKNN model to: {output_path}')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Failed to export RKNN model!')
        sys.exit(1)

    print('--> INT8 Conversion completed successfully!')
    rknn.release()

if __name__ == '__main__':
    workspace_dir = os.getenv('GITHUB_WORKSPACE', '.')

    default_onnx = os.path.join(workspace_dir, 'yolov8s.onnx')
    default_output = os.path.join(workspace_dir, 'output.rknn')
    default_dataset = os.path.join(workspace_dir, 'dataset.txt')

    onnx_file = sys.argv[1] if len(sys.argv) > 1 else default_onnx
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    target_platform = sys.argv[3] if len(sys.argv) > 3 else 'rk3588'
    dataset_file = sys.argv[4] if len(sys.argv) > 4 else default_dataset

    print(f"Using Workspace Directory: {workspace_dir}")
    convert_model(onnx_file, output_file, target_platform, dataset_file)
