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
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("Failed to load ONNX model.")
        exit(1)

    print("--> Building base graph for hybrid calibration...")
    ret = rknn.build(do_quantization=True, dataset='dataset.txt')
    if ret != 0:
        print("Build failed.")
        exit(1)

    # Correctly pass the configuration path to step 1
    cfg_path = rknn_path + ".quantization.cfg"
    print(f"--> Generating hybrid quantization config at: {cfg_path}")
    ret = rknn.hybrid_quantization_step1(cfg_path)
    if ret != 0:
        print("Hybrid quantization step 1 failed.")
        exit(1)

    print("Success! Hybrid configuration generated.")

if __name__ == '__main__':
    main()
