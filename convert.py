import sys
from rknn.api import RKNN

def convert_model(onnx_path, output_path, platform="rk3588", dataset_path=None):
    rknn = RKNN(verbose=True)

    print('--> Configuring model settings for Rock 5C (RK3588S)')
    rknn.config(import sys
from rknn.api import RKNN

def convert_model(onnx_path, rknn_path, target_platform):
    rknn = RKNN(verbose=True)
    print('--> Config model')
    rknn.config(target_platform=target_platform)
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX model failed!')
        sys.exit(1)
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        sys.exit(1)
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed!')
        sys.exit(1)
    print('Conversion completed successfully.')
    rknn.release()

if __name__ == '__main__':
    onnx_file = sys.argv[1]
    rknn_file = sys.argv[2]
    platform = sys.argv[3]
    convert_model(onnx_file, rknn_file, platform)

        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform=platform
    )

    print(f'--> Loading ONNX model: {onnx_path}')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Failed to load ONNX model!')
        exit(ret)

    print('--> Building RKNN model...')
    do_quant = dataset_path is not None
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    if ret != 0:
        print('Failed to build RKNN model!')
        exit(ret)

    print(f'--> Exporting RKNN model to: {output_path}')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Failed to export RKNN model!')
        exit(ret)

    print('--> Conversion complete!')
    rknn.release()

if __name__ == '__main__':
    onnx_file = sys.argv[1] if len(sys.argv) > 1 else 'model.onnx'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'output.rknn'
    ds_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_model(onnx_file, out_file, platform="rk3588", dataset_path=ds_path)
