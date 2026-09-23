import sys
from rknn.api import RKNN


def main():
  onnx_path = "ssd-mobilenet-v2.onnx"
  rknn_output_path = "ssd_mobilenet_v2_fp16.rknn"

  rknn = RKNN(verbose=True)

  print("--> Configuring NPU target...")
  rknn.config(
      mean_values=[[127.5, 127.5, 127.5]],
      std_values=[[127.5, 127.5, 127.5]],
      target_platform="rk3588",
  )

  print(f"--> Loading ONNX model: {onnx_path}")
  if rknn.load_onnx(model=onnx_path) != 0:
    print("Failed to load ONNX model.")
    sys.exit(1)

  print("--> Building FP16 model (No calibration needed)...")
  if rknn.build(do_quantization=False) != 0:
    print("Build failed.")
    sys.exit(1)

  print(f"--> Exporting RKNN model to {rknn_output_path}...")
  if rknn.export_rknn(rknn_output_path) != 0:
    print("Export failed.")
    sys.exit(1)

  print("Success! FP16 RKNN model built successfully.")
  rknn.release()


if __name__ == "__main__":
  main()
