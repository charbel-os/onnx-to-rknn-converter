import sys
from rknn.api import RKNN

ONNX_MODEL = 'yolov8s.onnx'
RKNN_MODEL = 'yolov8s_640_int8.rknn'
DATASET_TXT = 'dataset.txt'

def main():
    rknn = RKNN(verbose=True)

    # 1. Configure pre-processing / target platform & quantization settings
    print('--> Configuring model target and quantization parameters...')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rk3588',
        
        # --- FIXES & ENHANCEMENTS FOR YOLOv8 ---
        # Ultralytics YOLOv8 expects RGB. If your dataset.txt images are standard 
        # JPEGs/PNGs read/saved via OpenCV (which are BGR), set quant_img_RGB2BGR=True 
        # so the quantizer converts them to RGB. If they are already RGB, set to False.
        quant_img_RGB2BGR=True, 
        
        # Switch to kl_divergence to prevent severe accuracy/confidence drift 
        # compared to the default 'normal' algorithm.
        quantized_algorithm='kl_divergence',
        
        # Use channel-wise quantization for better weight distribution precision
        quantized_method='channel',
        
        # Keep full optimization enabled
        optimization_level=3
    )

    # 2. Load ONNX model
    print(f'--> Loading ONNX model: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Error: Load ONNX failed!')
        sys.exit(1)

    # 3. Build model with INT8 Quantization Enabled
    print('--> Building INT8 RKNN model (with calibration)...')
    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
    if ret != 0:
        print('Error: Build failed!')
        sys.exit(1)

    # 4. Export RKNN model
    print(f'--> Exporting RKNN model to {RKNN_MODEL}...')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Error: Export RKNN failed!')
        sys.exit(1)
    
    print('--> INT8 Quantization and Export completed successfully!')
    rknn.release()

if __name__ == '__main__':
    main()
