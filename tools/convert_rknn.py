import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import os

if __name__ == '__main__':
    model_path = './ttfnet.pt'
    input_size_list = [[1, 3, 512, 512]]

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[0, 0, 0],
                std_values=[255, 255, 255],
                quant_img_RGB2BGR=False,
                quantized_dtype="asymmetric_quantized-8",
                quantized_algorithm="normal",
                quantized_method="channel",
                float_dtype="float16",
                optimization_level=3,
                target_platform="rk3566",
                custom_string=None,
                remove_weight=False,
                compress_weight=False)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model_path, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./demo_list.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('./ttfnet.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img_path = './test/test.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    rknn.release()
