import sys
import os
from rknn.api import RKNN

def convert_model(onnx_path, output_path, platform="rk3588"):
    onnx_path = os.path.abspath(onnx_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(onnx_path):
        print(f"ERROR: Could not find ONNX model at: {onnx_path}")
        sys.exit(1)

    rknn = RKNN(verbose=True)

    print(f'--> Configuring model settings for platform: {platform}')
    # Standard normalization config for YOLOv8 (scales pixel inputs from 0-255 to 0-1)
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

    # INT8 Quantization requires a dataset.txt pointing to representative images
    dataset_path = 'dataset.txt'
    if not os.path.exists(dataset_path):
        print(f"ERROR: Quantization dataset file '{dataset_path}' is missing!")
        sys.exit(1)

    print(f'--> Building model as INT8 using dataset: {dataset_path}...')
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print('Failed to build RKNN INT8 model!')
        sys.exit(1)

    print(f'--> Exporting RKNN model to: {output_path}')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Failed to export RKNN model!')
        sys.exit(1)

    print('--> INT8 Quantization & Conversion completed successfully!')
    rknn.release()

if __name__ == '__main__':
    workspace_dir = os.getenv('GITHUB_WORKSPACE', '.')

    default_onnx = os.path.join(workspace_dir, 'yolov8s.onnx')
    default_output = os.path.join(workspace_dir, 'output.rknn')

    onnx_file = sys.argv[1] if len(sys.argv) > 1 else default_onnx
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    target_platform = sys.argv[3] if len(sys.argv) > 3 else 'rk3588'

    print(f"Using Workspace Directory: {workspace_dir}")
    convert_model(onnx_file, output_file, target_platform)
