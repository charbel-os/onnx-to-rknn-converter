import sys
from rknn.api import RKNN


def main():
    onnx_path = "yolov8n.onnx"
    rknn_output_path = "yolov8n_fp16.rknn"

    rknn = RKNN(verbose=True)

    print("--> Configuring NPU target for ROCK 5C (RK3588)...")
    rknn.config(
        target_platform="rk3588",
    )

    print(f"--> Loading ONNX model: {onnx_path}")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("Failed to load ONNX model.")
        sys.exit(1)

    print("--> Building FP16 model (No quantization)...")
    if rknn.build(do_quantization=False) != 0:
        print("Build failed.")
        sys.exit(1)

    print(f"--> Exporting RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("Export failed.")
        sys.exit(1)

    print("Success! YOLOv8n FP16 RKNN model built successfully.")
    rknn.release()


if __name__ == "__main__":
    main()
