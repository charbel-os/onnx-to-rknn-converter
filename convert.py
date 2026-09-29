    # 1. Professional Configuration for RK3588 NPUimport sys
from rknn.api import RKNN

ONNX_MODEL = 'yolov8s.onnx'
RKNN_MODEL = 'yolov8s_640_int8.rknn'
DATASET_TXT = 'dataset.txt'

def main():
    rknn = RKNN(verbose=True)

    # 1. Professional Configuration for RK3588 NPU
    print('--> Configuring high-performance target and quantization parameters...')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rk3588',
        quant_img_RGB2BGR=True,
        
        # High-tier accuracy settings supported by RKNN-Toolkit2 v2.3.2
        quantized_algorithm='mmse',      # Minimizes quantization loss for better bounding box confidence
        quantized_method='channel',      # Channel-wise precision distribution
        optimization_level=3             # Full graph optimization (Conv+BN fusion, dead node stripping)
    )

    # 2. Load ONNX model
    print(f'--> Loading ONNX model: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Error: Load ONNX failed!')
        sys.exit(1)

    # 3. Build model with High-Precision INT8 Calibration via your dataset.txt
    print(f'--> Building INT8 RKNN model using calibration dataset: {DATASET_TXT}...')
    ret = rknn.build(
        do_quantization=True, 
        dataset=DATASET_TXT,
        pre_compile=True                     # Pre-compiles graph layout for fast runtime initialization on board
    )
    if ret != 0:
        print('Error: Build failed!')
        sys.exit(1)

    # 4. Export RKNN model
    print(f'--> Exporting production RKNN model to {RKNN_MODEL}...')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Error: Export RKNN failed!')
        sys.exit(1)
    
    print('--> Production INT8 Optimization and Export completed successfully!')
    rknn.release()

if __name__ == '__main__':
    main()
