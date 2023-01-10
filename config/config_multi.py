import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")

from public.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize, RandomAffine, RandomColorAndBlur

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    '''************************************************************'''
    '''Multi Task params'''
    #the num of the heads, 2 recommond
    head_num = 2
    #choose the head you want to train
    train_head_pos = 2 #in [1, 2, ...head_num]
    #set the param to be True, when you want to deploy the model
    deploy = False

    '''************************************************************'''
    task_name = "Multi_task"
    version = 2
    #the type of network: FCOS, CenterNet.... in public.models
    network = "CenterNetMul"
    #the type of backbone: resnet18, resnet50, swin_t, swin_b, densecl_resnet50_coco...  in public.backbone
    backbone_type = "resnet50"                                                   
    # Path to save log
    log = '../logs/{}/{}/log_v{}'.format(task_name, network+"_"+backbone_type , version)  
    # Path to store checkpoint model
    checkpoint_path = '../checkpoints/{}/{}/v{}'.format(task_name, network+"_"+backbone_type, version)  
    # load checkpoint model
    resume = checkpoint_path + '/latest.pth'

    #the model trained by previous task
    '''for example, when I am training task2, I should use the model trained in pervious version,
    pre_model_dir='../checkpoints/{}/{}/v{}'.format(task_name, network+"_"+backbone_type, version-1)'''
    pre_model_dir = '../checkpoints/{}/{}/v{}/latest.pth'.format(task_name, network+"_"+backbone_type, version-1)

    '''************************************************************'''
    '''training details'''
    input_image_size = 512
    num_classes = [15,80]
    assert len(num_classes) == head_num, "len(num_classes) equal head_num"
    epochs = 140
    per_node_batch_size = 16
    lr = 1e-4
    num_workers = 8
    print_interval = 100
    eval_interval = 1
    apex = False
    sync_bn = True

    '''************************************************************'''
    '''backbone'''
    #use the pretrained backbone
    pretrained = True
    if pre_model_dir:
        pretrained = False
    if "swin" in backbone_type:
        #if you want to use TTF neck or deploy, you should set out_indices=(0,1,2,3)
        backbone_dict = dict(out_indices=(3,))
    else:
        backbone_dict = None
    '''************************************************************'''
    '''neck'''
    #the type of neck: CenterNetNeck, TTF, RetinaFPN_TransConv, RetinaFPN, YolofDC5... in public.neck

    #use the param when you use normal fpn as neck
    #use the p5 to downsample (set to be True common)
    neck_dict1 = dict(neck_type = "CenterNetNeck",
                    param=dict(num_layers=3,
                        out_channels=[256, 128, 64]))
    neck_dict2 = dict(neck_type = "TTFNeck",
                    param=dict(out_channels=[256, 128, 64],selayer=True))
    neck_dicts = []
    loc = locals()
    for i in range(1,head_num+1):
        s = "neck_dict" + str(i)
        try:
            cur_neck = loc[s]
        except:
            raise ValueError(f"neck_dict{i} didn't exist!! please add!!")
        neck_dicts.append(cur_neck)
    if (neck_dicts[train_head_pos-1]["neck_type"] == "TTFNeck" or deploy) and "swin" in backbone_type:
        backbone_dict["out_indices"] = (0,1,2,3)

    #use the param when you use yolof as neck
    # if "Yolof" in neck_type:
    #     neck_out_channles = 512
    #     neck_dict = dict(encoder_channels=neck_out_channles,
    #                     block_mid_channels=128,
    #                     num_residual_blocks=4,
    #                     block_dilations=[4, 8, 12, 16])


    '''************************************************************'''
    '''Head'''
    head_dict1 = dict(num_classes=num_classes[0],
                 out_channels=64)
    head_dict2 = dict(num_classes=num_classes[1],
                 out_channels=64)

    head_dicts = []
    loc = locals()
    for i in range(1,head_num+1):
        s = "head_dict" + str(i)
        try:
            cur_head = loc[s]
        except:
            raise ValueError(f"head_dict{i} didn't exist!! please add!!")

        head_dicts.append(cur_head)

    '''************************************************************'''
    '''decoder'''
    #down sample strides
    decoder_dict = dict(image_w=input_image_size,
                        image_h=input_image_size,
                        min_score_threshold=0.05,
                        max_detection_num=100,
                        topk=100,
                        stride=4)


    '''************************************************************'''
    '''Loss'''
    '''CenternetLoss''' #set the param in different network
    loss_dict = dict(
       alpha=2.,
       beta=4.,
       wh_weight=0.1,
       epsilon=1e-4,
       min_overlap=0.7,
       max_object_num=100
    )


    '''************************************************************'''
    '''Dataset'''
    # evaluate model path, fill the path if you only want to evaluate your model
    ''''you must set you file format as coco2017, if you want program to automatically choose the current training dataset!!'''
    evaluate = None
    base_path = ['/home/jovyan/data-vol-polefs-1/dataset/flags/coco/',
                 '/home/jovyan/data-vol-polefs-1/dataset/coco/']
    
    train_dataset_path = os.path.join(base_path[train_head_pos-1], 'train2017')
    val_dataset_path = os.path.join(base_path[train_head_pos-1], 'val2017')
    dataset_annotations_path = os.path.join(base_path[train_head_pos-1], 'annotations')
    
    
    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train2017",
                                  transform=transforms.Compose([
                                      RandomColorAndBlur(),
                                    #   RandomAffine(),
                                      RandomFlip(flip_prob=0.3),
                                      RandomCrop(crop_prob=0.3),
                                    #   RandomTranslate(translate_prob=0.3),
                                      Normalize(),
                                      Resize(resize=input_image_size)
                                  ]))
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val2017",
                                transform=transforms.Compose([
                                    Normalize(),
                                    Resize(resize=input_image_size),
                                ]))
    
***********************************************************#



    
