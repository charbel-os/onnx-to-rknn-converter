import sys
import os
from ultralytics import YOLO
from rknn.api import RKNN

ONNX_MODEL = 'yolov8n-seg.onnx'
RKNN_MODEL = 'yolov8n_seg_640_int8.rknn'
DATASET_TXT = 'dataset.txt'

def generate_dataset_txt():
    """Generates the dataset.txt file listing image paths for RKNN INT8 calibration."""
    print('--> Generating dataset.txt for INT8 calibration...')
    image_dir = 'coco128/images/train2017'
    
    if not os.path.exists(image_dir):
        print(f"Error: Calibration image directory '{image_dir}' not found!")
        sys.exit(1)
        
    images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.jpeg', '.png'))][:100]
    
    with open(DATASET_TXT, 'w') as f:
        for img_path in images:
            f.write(f"{img_path}\n")
    print(f'--> Created {DATASET_TXT} with {len(images)} calibration images.')

def main():
    # Step A: Export YOLOv8n-seg PyTorch to ONNX first
    print('--> Exporting YOLOv8n-seg PyTorch model to ONNX...')
    model = YOLO('yolov8n-seg.pt')
    model.export(format='onnx', imgsz=640, simplify=True, dynamic=False)

    if not os.path.exists(ONNX_MODEL):
        # Fallback renaming if ultralytics names it slightly differently
        if os.path.exists('yolov8n-seg.onnx'):
            pass
        else:
            print("Error: ONNX export failed to generate 'yolov8n-seg.onnx'.")
            sys.exit(1)

    # Step B: Generate the dataset text file needed for RKNN INT8 calibration
    generate_dataset_txt()

    # Step C: Initialize RKNN API
    rknn = RKNN(verbose=True)

    # 1. Configure pre-processing / target platform
    print('--> Configuring model target for RK3588...')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform='rk3588',
        quantized_algorithm='kl_divergence'
    )

    # 2. Load ONNX model
    print(f'--> Loading ONNX model into RKNN: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load ONNX failed!')
        sys.exit(1)

    # 3. Build model with INT8 Quantization Enabled
    print('--> Building INT8 RKNN segmentation model (with calibration)...')
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

    print('--> INT8 Segmentation Quantization and Export completed successfully!')
    rknn.release()

if __name__ == '__main__':
    main()
