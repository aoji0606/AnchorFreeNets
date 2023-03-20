import os
import sys

sys.path.append("../")

import torch
import time
from tqdm import tqdm
from public import models
# from config.config_ttfnet import Config
from onnx import load_model, save_model
from onnxmltools.utils import float16_converter
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    # model = models.__dict__[Config.network](**{
    #     "pretrained": Config.pretrained,
    #     "backbone_type": Config.backbone_type,
    #     "neck_type": Config.neck_type,
    #     "neck_dict": Config.neck_dict,
    #     "head_dict": Config.head_dict,
    #     "backbone_dict": Config.backbone_dict
    # })
    #
    # model_dir = os.path.join(Config.checkpoint_path, "best.pth")
    # checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint, strict=False)
    #
    # model.eval()
    # model.cuda()
    #
    # x = torch.randn([1, 3, 512, 512]).cuda()
    # with torch.no_grad():
    #     torch.onnx.export(
    #         model,
    #         x,
    #         "./model_fp32.onnx",
    #         input_names=["input"],
    #         output_names=["output"],
    #         opset_version=11,
    #         dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    #         verbose=False
    #     )
    #
    # onnx_model = load_model("./model_fp32.onnx")
    # trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
    # save_model(trans_model, "model_fp16.onnx")

    # onne run
    x = torch.randn([1, 3, 512, 512])
    model = onnxruntime.InferenceSession("model_fp16.onnx", None,
                                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    loop = 5000
    t = 0.0
    for i in tqdm(range(loop)):
        t1 = time.time()
        inputs = {model.get_inputs()[0].name: to_numpy(x)}
        heatmap_output, hw_output = model.run(None, inputs)
        t2 = time.time()

        t += (t2 - t1)

    print(t * 1000 / loop)
