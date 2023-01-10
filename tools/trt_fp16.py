import torch.nn as nn
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import sys
import time
from collections import OrderedDict

sys.path.append("../")

from public import models, decoder
from config.config_fcos import Config
from torch2trt import torch2trt
#from torchvision.models.resnet50 import alexnet
# import torchvision.models as models
from torch2trt import TRTModule

if __name__ == "__main__":
    torch.cuda.set_device(0)
    # torch.backends.cudnn.benchmark = True
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())   
    resize = Config.input_image_size
    device = torch.device("cpu")
    model_dir = os.path.join(Config.checkpoint_path, "best.pth")

    model = models.__dict__[Config.network](**{
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type, 
        "neck_type": Config.neck_type, 
        "neck_dict": Config.neck_dict, 
        "head_dict": Config.head_dict, 
        "backbone_dict": Config.backbone_dict
    })

    model.eval().cuda()

    # create example data
    x = torch.ones((8, 3, 512, 512)).cuda()

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(model, [x], fp16_mode=False, max_batch_size=8, use_onnx=True)
    torch.save(model_trt.state_dict(), 'resnet50_trt.pth')
    model_trt = torch2trt(model, [x], fp16_mode=True, max_batch_size=8, use_onnx=True)
    torch.save(model_trt.state_dict(), 'resnet50_trt_fp16.pth')
    print ("hhhh")

    
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('resnet50_trt.pth'))
    model_trt_fp16 = TRTModule()
    model_trt_fp16.load_state_dict(torch.load('resnet50_trt_fp16.pth'))

    x = torch.ones((8, 3, 512, 512)).cuda().contiguous()
    # print ("test pytorch")
    # stime = time.time()
    # with torch.no_grad():
    #     for i in range(1000):
    #         torch.cuda.synchronize()
    #         res= model(x)
    #         torch.cuda.synchronize()
    # time1 =  time.time() - stime
    # print ("time = ", time1)

    # torch.cuda.empty_cache()
    # model.half()
    # x = x.half()
    # print ("test pytorch half")
    # stime = time.time()
    # with torch.no_grad():
    #     for i in range(1000):
    #         torch.cuda.synchronize()
    #         res= model(x)
    #         torch.cuda.synchronize()
    # time1 =  time.time() - stime
    # print ("time = ", time1)


    print ("test trt")
    stime = time.time()
    for i in range(125):
        torch.cuda.synchronize()
        res= model_trt(x)
        torch.cuda.synchronize()
    time2 =  time.time() - stime
    print ("time = ", time2)

    print ("test trt fp16")
    stime = time.time()
    for i in range(125):
        torch.cuda.synchronize()
        res= model_trt_fp16(x)
        torch.cuda.synchronize()
    time3 =  time.time() - stime
    print ("time = ", time3)
    
    
