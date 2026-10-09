import sys
import os
from ultralytics import YOLO
from rknn.api import RKNN

ONNX_MODEL = 'yolov8n-seg.onnx'
RKNN_MODEL = 'yolov8n_seg_640_int8.rknn'
DATASET_TXT = 'dataset.txt'

def generate_dataset_txt():
    """Generates the dataset.txt file listing image paths for RKNN INT8 calibration (200 images)."""
    print('--> Generating dataset.txt for INT8 calibration...')
    possible_dirs = [
        'coco128/images/train2017',
        'coco128/images',
        'datasets/coco128/images/train2017'
    ]
    
    image_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            image_dir = d
            break
            
    if not image_dir:
        print(f"Error: Calibration image directory not found in expected paths: {possible_dirs}")
        sys.exit(1)
        
    print(f"--> Using image directory: {image_dir}")
    # Increased calibration images limit to 200
    images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:200]
    
    if not images:
        print(f"Error: No images found in {image_dir}!")
        sys.exit(1)
    
    with open(DATASET_TXT, 'w') as f:
        for img_path in images:
            f.write(f"{img_path}\n")
    print(f'--> Created {DATASET_TXT} with {len(images)} calibration images.')

def main():
    # Step A: Export YOLOv8n-seg PyTorch to ONNX first
    print('--> Exporting YOLOv8n-seg PyTorch model to ONNX...')
    model = YOLO('yolov8n-seg.pt')
    model.export(format='onnx', imgsz=640, simplify=True, dynamic=False, opset=12)
    
    if not os.path.exists(ONNX_MODEL):
        print("Error: ONNX export failed to generate 'yolov8n-seg.onnx'.")
        sys.exit(1)

    # Step B: Generate calibration dataset text file (200 images)
    generate_dataset_txt()

    # Step C: Initialize RKNN API
    rknn = RKNN(verbose=True)

    # 1. Configure pre-processing and target platform for full INT8 quantization
    print('--> Configuring model target for RK3588 (Full INT8)...')
    config_kwargs = {
        'mean_values': [[0, 0, 0]], 
        'std_values': [[255, 255, 255]], 
        'target_platform': 'rk3588',
        'quantized_algorithm': 'kl_divergence',
        'quantized_dtype': 'asymmetric_quantized-u8',
        'optimization_level': 3
    }

    rknn.config(**config_kwargs)

    # 2. Load ONNX model
    print(f'--> Loading ONNX model into RKNN: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load ONNX failed!')
        sys.exit(1)

    # 3. Build model with full INT8 Quantization Enabled
    print('--> Building Full INT8 RKNN segmentation model (with calibration)...')
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

    print('--> Full INT8 Segmentation Quantization and Export completed successfully!')
    rknn.release()

if __name__ == '__main__':
    main()
