import os
import sys
import torch

sys.path.append("../")

from public import models
from config.config_ttfnet import Config

if __name__ == '__main__':
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
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)

    torch.save(model, os.path.join(Config.checkpoint_path, "teacher.pth"))
