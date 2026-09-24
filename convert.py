import sys
from rknn.api import RKNN
from ultralytics import YOLO


def main():
    img_size = 412
    # YOLOv8s exports to yolov8s.onnx by default when using yolov8s.pt
    onnx_path = f"yolov8s_{img_size}.onnx"
    rknn_output_path = f"yolov8s_{img_size}_fp16.rknn"

    print(
        f"--> Step 1: Training/Fine-tuning YOLOv8s (imgsz={img_size})..."
    )
    # Using 'yolov8s.pt' base weights and lightweight coco8.yaml dataset
    model = YOLO("yolov8s.pt")
    model.train(data="coco8.yaml", epochs=1, imgsz=img_size, batch=4)

    print(f"--> Step 2: Exporting trained model to ONNX (imgsz={img_size})...")
    # Ultralytics appends the size if specified or renames; we ensure correct naming
    model.export(format="onnx", imgsz=img_size, simplify=True)

    print("--> Step 3: Configuring NPU target for ROCK 5C (RK3588)...")
    rknn = RKNN(verbose=True)
    rknn.config(target_platform="rk3588")

    print(f"--> Step 4: Loading ONNX model: {onnx_path}")
    if rknn.load_onnx(model=onnx_path) != 0:
        print("Failed to load ONNX model.")
        sys.exit(1)

    print("--> Step 5: Building FP16 RKNN model...")
    if rknn.build(do_quantization=False) != 0:
        print("Build failed.")
        sys.exit(1)

    print(f"--> Step 6: Exporting RKNN model to {rknn_output_path}...")
    if rknn.export_rknn(rknn_output_path) != 0:
        print("Export failed.")
        sys.exit(1)

    print(
        f"Success! YOLOv8s (imgsz={img_size}) FP16 RKNN model built successfully."
    )
    rknn.release()


if __name__ == "__main__":
    main()
