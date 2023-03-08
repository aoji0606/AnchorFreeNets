import torch
import os
import sys

sys.path.append("../")

from public import models
from config.config_ttfnet import Config

if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cpu")

    model = models.__dict__[Config.network](**{
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type,
        "neck_type": Config.neck_type,
        "neck_dict": Config.neck_dict,
        "head_dict": Config.head_dict,
        "backbone_dict": Config.backbone_dict
    })

    model_dir = os.path.join(Config.checkpoint_path, "best.pth")
    if os.path.exists(model_dir):
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    trace_model = torch.jit.trace(model, torch.Tensor(1, 3, Config.input_image_size[0], Config.input_image_size[1]))
    trace_model.save("/home/jovyan/data-vol-polefs-1/rk3566/torch2rknn/ttfnet.pt")
    print("success save trace model")
