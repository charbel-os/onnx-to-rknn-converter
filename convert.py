import sys
from rknn.api import RKNN

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 convert.py <onnx_path> <rknn_path> <target_platform>")
        sys.exit(1)

    onnx_path = sys.argv[1]
    rknn_path = sys.argv[2]
    target_platform = sys.argv[3]

    rknn = RKNN(verbose=False)

    print("--> Configuring model settings...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform
    )

    print(f"--> Loading ONNX model: {onnx_path}")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("Failed to load ONNX model.")
        sys.exit(1)

    print("--> Building model with pure INT8 quantization...")
    if rknn.build(do_quantization=True, dataset='dataset.txt') != 0:
        print("Build failed.")
        sys.exit(1)

    print(f"--> Exporting RKNN model to: {rknn_path}")
    if rknn.export_rknn(rknn_path) != 0:
        print("Export failed.")
        sys.exit(1)

    print("Success! YOLOv5 INT8 model created.")

if __name__ == '__main__':
    main()
