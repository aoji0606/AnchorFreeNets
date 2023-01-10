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
    task_name = "flag"
    version = 1
    #the type of network: FCOS, CenterNet.... in public.models
    network = "CenterNet"
    #the type of backbone: resnet18, resnet50, swin_t, swin_b...  in public.backbone
    backbone_type = "resnet50"
    # Path to save log
    log = '../logs/{}/{}/log_v{}'.format(task_name, network+"_"+backbone_type , version)  
    # Path to store checkpoint model
    checkpoint_path = '../checkpoints/{}/{}/v{}'.format(task_name, network+"_"+backbone_type, version)  
    # load checkpoint model
    resume = checkpoint_path + '/latest.pth'


    input_image_size = 512
    num_classes = 15
    epochs = 140
    per_node_batch_size = 8
    lr = 1e-4
    num_workers = 4
    print_interval = 100
    eval_interval = 1
    apex = False
    sync_bn = True

    '''************************************************************'''
    '''backbone'''
    #use the pretrained backbone
    pretrained = True
    if "swin" in backbone_type:
        backbone_dict = dict(out_indices=(3,))
    else:
        backbone_dict = None
    '''************************************************************'''
    '''neck'''
    #the type of neck: CenterNetNeck, TTFNeck, YolofDC5, RetinaFPN_TransConv, RetinaFPN... in public.neck
    #if you want to use fpn, you must set the neck_dict by yourself!!!!!!!!!!!!!!!
    neck_type = "TTFNeck"

    #use the param when you use normal fpn as neck
    #use the p5 to downsample (set to be True common)
    neck_out_channles = 64
    neck_dict = dict(num_layers=3,
                    out_channels=[256, 128, 64])

    #use the param when you use yolof as neck
    if "Yolof" in neck_type:
        if "swin" in backbone_type:
            backbone_dict = dict(out_indices=(2,3))
        neck_out_channles = 512
        neck_dict = dict(dila_dict=dict(encoder_channels=neck_out_channles,
                        block_mid_channels=128,
                        num_residual_blocks=4,
                        block_dilations=[4, 8, 12, 16]))
    
    #use the param when you use TTF as neck
    if "TTF" in neck_type:
        if "swin" in backbone_type:
            backbone_dict = dict(out_indices=(0,1,2,3))
        neck_dict = dict(out_channels=[256, 128, 64], selayer=False)


    '''************************************************************'''
    '''Head'''
    head_dict = dict(num_classes=num_classes,
                 out_channels=neck_out_channles)

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
    '''CenternetLoss'''
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
    #the pretrained model dir (not the backbone)
    pre_model_dir = None
    # evaluate model path, fill the path if you only want to evaluate your model
    evaluate = None
    base_path = '/home/jovyan/data-vol-polefs-1/dataset/flags/coco/'
    train_dataset_path = os.path.join(base_path, 'train2017')
    val_dataset_path = os.path.join(base_path, 'val2017')
    dataset_annotations_path = os.path.join(base_path, 'annotations')
    
    
    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train2017",
                                  transform=transforms.Compose([
                                      RandomColorAndBlur(),
                                      RandomAffine(),
                                      RandomFlip(flip_prob=0.3),
                                      RandomCrop(crop_prob=0.3),
                                      RandomTranslate(translate_prob=0.3),
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
    




    
