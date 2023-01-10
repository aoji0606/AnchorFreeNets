import torch
import os
import sys
import time
from torch2trt import torch2trt
from torch2trt import TRTModule

sys.path.append("../")
from public import models
from config.config_centernet import Config


def get_input(device):
    return torch.ones([1, 3, kInputSize, kInputSize]).to(device)


def convert_to_tensorRT(model, pt_path, trt_path):
    heads = {}
    heads['hm'] = 14
    heads['wh'] = 2
    heads['reg'] = 2
    head_conv = 64
    down_ratio = 4

    model_path = pt_path
    model_trt_path = trt_path
    print(model_path)

    assert (os.path.exists(model_path))

    start = time.time()
    model.load_state_dict(torch.load(model_path))
    model = model.eval().cuda()
    stop = time.time()
    print('load torch model cost %.1f ms' % ((stop - start) * 1000.0), 'ms')

    x = torch.ones((8, 3, kInputSize, kInputSize)).cuda()
    start = time.time()
    model_trt = torch2trt(model, [x], max_batch_size=8, float16=True)
    stop = time.time()
    print('convert to tensorRT cost %.1f seconds' % (stop - start))
    # assert(0)

    start_trt = time.time()
    torch.save(model_trt.state_dict(), model_trt_path)
    end_trt = time.time()
    print('save tensorrt model cost %.1f seconds' % (end_trt - start_trt))


def test_trt(trt_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    model = TRTModule()
    if not os.path.exists(trt_path):
        print('Can not find %s' % (trt_path))
        return
    start = time.time()
    model.load_state_dict(torch.load(trt_path))
    stop = time.time()
    print('load trt model cost %.1f ms' % ((stop - start) * 1000.0))

    # read image
    img = get_input(device)

    # forward
    feature = model(img)
    start = time.time()
    kTimes = 1
    for i in range(kTimes):
        feature = model(img)
    stop = time.time()
    print('forward average cost: %.3f ms' % ((stop - start) * 1000.0 / kTimes))


if __name__ == "__main__":
    kInputSize = 512
    use_cuda = True
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

    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    if use_cuda:
        device = torch.device("cuda:0")
        model = model.to(device)
    x = torch.ones((8, 3, kInputSize, kInputSize)).cuda()
    start = time.time()
    model_trt = convert_to_tensorRT(model, )
    stop = time.time()
    print('convert to tensorRT cost %.1f seconds' % (stop - start))
