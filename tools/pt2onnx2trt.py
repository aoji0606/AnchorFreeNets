import torch
from torch2trt import torch2trt
import sys

sys.path.append("../")

from public import models
from config.config_fcos import Config
from onnx import ModelProto
import tensorrt as trt


def build_engine(onnx_path, shape=[1, 3, 512, 512]):
    with trt.Builder(TRT_LOGGER) as \
            builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as \
            network, trt.OnnxParser(network, TRT_LOGGER) as \
            parser:
        builder.max_workspace_size = 1 << 25
        builder.fp16_mode = True
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


if __name__ == "__main__":
    # create some regular pytorch model...
    model = models.__dict__[Config.network](**{
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type,
        "neck_type": Config.neck_type,
        "neck_dict": Config.neck_dict,
        "head_dict": Config.head_dict,
        "backbone_dict": Config.backbone_dict
    })

    # pt_model_path = sys.argv[1]
    onnx_model_path = "fcos.protof"
    engine_path = 'fcos.engine'
    # print("name ", pt_model_path)
    # test_model = torch.load(pt_model_path, map_location=torch.device('cpu'))

    # model.load_state_dict(test_model, strict = False)
    model.eval().cuda()  # float16

    # create example data
    x = torch.ones((8, 3, 512, 512)).cuda()

    with torch.no_grad():
        torch.onnx.export(model, x, onnx_model_path, verbose=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    batch_size = 8

    model = ModelProto()
    with open(onnx_model_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size, d0, d1, d2]
    print(shape)
    engine = build_engine(onnx_model_path, shape=shape)
    save_engine(engine, engine_path)

# from torch2trt import TRTModule
# import torch
# import tensorrt as trt
# import pycuda.driver as cuda

# trt_logger = trt.Logger(trt.Logger.INFO)

# with open("fcos.engine", 'rb') as f, trt.Runtime(trt_logger) as runtime:
#     model = runtime.deserialize_cuda_engine(f.read())

# input_name = ["input"]
# output_name = ["output"+ str(i) for i in range(15)]
# trt_model = TRTModule(engine=model)
# x = torch.randn((1,3,512,512)).cuda()
# out = trt_model(x)
# for item in out:
#     print(out.shape)
