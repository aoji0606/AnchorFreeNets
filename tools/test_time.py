import sys
import time
from tqdm import tqdm

import torch

sys.path.append("../")

from public import models
from config.config_ttfnet_pruning import Config

if __name__ == "__main__":
    B = 32
    x = torch.randn(B, 3, 512, 512).cuda()

    model = models.__dict__[Config.network](**{
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type,
        "neck_type": Config.neck_type,
        "neck_dict": Config.neck_dict,
        "head_dict": Config.head_dict,
        "backbone_dict": Config.backbone_dict
    })
    print({
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type,
        "neck_type": Config.neck_type,
        "neck_dict": Config.neck_dict,
        "head_dict": Config.head_dict,
        "backbone_dict": Config.backbone_dict
    })
    # checkpoint = torch.load('/home/jovyan/data-vol-polefs-1/ttfnet_rmnet18_sr1e-4_voc.pth', map_location=torch.device('cpu'))
    checkpoint = torch.load(
        '/home/jovyan/data-vol-polefs-1/code/AnchorFreeNets/checkpoints/ttfnet/TTFNetPruning_rmnet_pruning_18/v1.0.0/best.pth',
        map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)

    # model.prune_by_thresh(0.1)
    model.prune(0.5)

    print(model)

    model = model.cuda()

    loop = 20 * 4
    for i in tqdm(range(loop)):
        y = model(x)

    loop = 100 * 4
    res = 0
    for i in tqdm(range(loop)):
        t1 = time.time()
        torch.cuda.synchronize()
        y = model(x)
        torch.cuda.synchronize()
        t2 = time.time()
        res += (t2 - t1)

    print((res / loop) * 1000 / B)
