import os

path = os.path.dirname(os.path.abspath(__file__))
pretrained_models_path = os.path.join(path, 'pretrained_models/')
print("auto create the pretrained model dir.........................")
os.mkdir(pretrained_models_path)

url = "http://ai-zzzc.s3.360.cn/model/AnchorFreeNet/pretrained_models/"
resnet = ['resnet50-epoch100-acc76.512.pth', 
        'resnet18-epoch100-acc70.316.pth', 
        'resnet50_half-epoch100-acc72.066.pth', 
        'resnet34_half-epoch100-acc67.472.pth', 
        'resnet152-epoch100-acc78.564.pth', 
        'densecl_r50_coco_1600ep.pth', 
        'resnet50_fcos_coco_resize667_mAP0.321.pth', 
        'densecl_r50_imagenet_200ep.pth', 
        'resnet34-epoch100-acc73.736.pth', 
        'resnet101-epoch100-acc77.724.pth']
swin = ['swin_small_patch4_window7_224.pth', 
        'swin_large_patch4_window7_224_22k.pth', 
        'swin_tiny_patch4_window7_224.pth', 
        'swin_base_patch4_window7_224_22k.pth']

all_item = []
respath = os.path.join(pretrained_models_path, "resnet")
swinpath = os.path.join(pretrained_models_path, "swin")
for item in resnet:
    cur = url + "resnet/" + item
    print(f"runing wget -P {respath} {cur}")
    os.system(f"wget -P {respath} {cur}")
    print(f"down {item} done!!")
for item in swin:
    cur = url + "swin/" + item
    print(f"runing wget -P {swinpath} {cur}")
    os.system(f"wget -P {swinpath} {cur}")
    print(f"down {item} done!!")
print("all done!!")