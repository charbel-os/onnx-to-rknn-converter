import sys
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
