# AnchorFreeNet

## get start

### environment

You should prepare the TensorRT>=7.0 and cuda>=10.0 environment!

``` shell
pip install -r requirement.txt
# or
# find torch version in https://pytorch.org/get-started/previous-versions/, e.g.
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pycocotools==2.0.2 timm==0.4.9 numpy fvcore Cython matplotlib opencv-python tqdm thop torchstat
```

then, you should run this commond to download the pretrained models!

```shell
python down_pretrained_model.py
```

***

## Modify your own config

### supported backbone

resnet, swin-transformer, regnet....

### supported Net

FCOS, CenterNet, TTFNet

***

## Train your own data and Test

**Introduction based on CenterNet!**

### modify the config file

Firstly, modify your dataset file frame

```shell
COCO2017
|
|-----annotations----instances_train2017(or val).json
|                 
|-----train2017
|-----val2017
```

modify the data_path and other necessary params in the config file([config_centernet.py])

```shell
base_path = '/your coco frame data path/'
```
***

## Train
```shell
cd ./train
./train.sh
```
***

## Test
```shell
cd ./tools
python eval.py
```
***

## Inference
```shell
cd ./tools
python inference.py
```
***

## Json Analysis
show per class metrics, such as: ap precision recall f1

save FP FN images
```shell
cd ./tools
python json_analysis.py
```
***

## Deploy

convert pytorch model to TRT model or RK model

tensorrt: trt_fp16.py  pt2trt.py  pt2onnx2trt.py

rk: export_trace_model.py  convert_rknn.py