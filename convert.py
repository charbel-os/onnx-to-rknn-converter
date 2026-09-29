import sys
from rknn.api import RKNN

ONNX_MODEL = 'yolov8s.onnx'
RKNN_MODEL = 'yolov8s_640_int8.rknn'
DATASET_TXT = 'dataset.txt'

def main():
    rknn = RKNN(verbose=True)

    # 1. Configure pre-processing / target platform
    print('--> Configuring model target...')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rk3588'
    )

    # 2. Load ONNX model
    print(f'--> Loading ONNX model: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load ONNX failed!')
        sys.exit(1)

    # 3. Build model with INT8 Quantization Enabled
    print('--> Building INT8 RKNN model (with calibration)...')
    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
    if ret != 0:
        print('Build failed!')
        sys.exit(1)

    # 4. Export RKNN model
    print(f'--> Exporting RKNN model to {RKNN_MODEL}...')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export RKNN failed!')
        sys.exit(1)
    
    print('--> INT8 Quantization and Export completed successfully!')
    rknn.release()

if __name__ == '__main__':
    main()
