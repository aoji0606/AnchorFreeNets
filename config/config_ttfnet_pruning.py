import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")

from public.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize, RandomAffine, RandomColorAndBlur

import torchvision.transforms as transforms


class Config(object):
    task_name = "ttfnet"
    version = "1.0.0"
    
    # the type of network: FCOS, CenterNet.... in public.models
    network = "TTFNetPruning"

    # prune backbone:
    # rmnet_pruning_18, rmnet_pruning_34
    # repvgg_pruning_a0, repvgg_pruning_a1, repvgg_pruning_a2, repvgg_pruning_b0, ...
    backbone_type = "rmnet_pruning_18"

    # Path to save log
    log = '../logs/{}/{}/log_v{}'.format(task_name, network+"_"+backbone_type , version)  
    # Path to store checkpoint model
    checkpoint_path = '../checkpoints/{}/{}/v{}'.format(task_name, network+"_"+backbone_type, version)  
    # load checkpoint model
    resume = checkpoint_path + '/latest.pth'

    num_classes = 20 # Note: The number of categories needs to be changed after changing the data set.

    # 2x
    epochs = 24
    milestones = [18, 22]

    # # 3x
    # epochs = 36
    # milestones = [24, 33]

    # # 10x
    # epochs = 120
    # milestones = [90, 110]

    input_image_size = 512
    per_node_batch_size = 16
    lr = 8e-3 / 20  # adamw lr
    warmup_epochs = 2
    warmup_ratio = 0.2
    num_workers = 6
    print_interval = 50
    eval_interval = 1
    apex = False
    sync_bn = True

    '''************************************************************'''
    '''backbone'''
    #use the pretrained backbone
    pretrained = True
    backbone_dict = dict(out_indices=(3,)) if 'swin' in backbone_type else None

    '''************************************************************'''
    '''neck'''
    #the type of neck: CenterNetNeck, TTFNeck, YolofDC5, RetinaFPN_TransConv, RetinaFPN... in public.neck
    #if you want to use fpn, you must set the neck_dict by yourself!!!!!!!!!!!!!!!
    neck_type = "TTFNeckPruning"

    # use the param when you use normal fpn as neck
    # use the p5 to downsample (set to be True common)
    neck_out_channles = 64
    neck_dict = dict(num_layers=3, out_channels=[256, 128, 64])

    #use the param when you use yolof as neck
    if "Yolof" in neck_type:
        if "swin" in backbone_type:
            backbone_dict = dict(out_indices=(2, 3))

        neck_out_channles = 512
        neck_dict = dict(
            dila_dict=dict(
                encoder_channels=neck_out_channles,
                block_mid_channels=128,
                num_residual_blocks=4,
                block_dilations=[4, 8, 12, 16]
            )
        )
    
    #use the param when you use TTF as neck
    if "TTF" in neck_type:
        if "swin" in backbone_type:
            backbone_dict = dict(out_indices=(0, 1, 2, 3))

        neck_dict = dict(out_channels=[256, 128, 64], selayer=False, deformable=True)


    '''************************************************************'''
    '''Head'''
    head_dict = dict(
        num_classes=num_classes,
        out_channels=neck_out_channles
    )

    '''************************************************************'''
    '''decoder'''
    # down sample strides
    decoder_dict = dict(
        image_w=input_image_size,
        image_h=input_image_size,
        min_score_threshold=0.01, # origin: 0.05,
        max_detection_num=100,
        topk=100,
        stride=4
    )


    '''************************************************************'''
    '''Loss'''
    '''CenternetLoss'''
    loss_dict = dict(
       focal_alpha=2.,
       focal_beta=4.,  # 5.,
       hm_weight=1.,
       wh_weight=5.,   # 0.1,
       epsilon=1e-4,
       gaussian_radius_alpha=0.54, # radius for ttfnet
       min_overlap=0.7,  # radius for centernet, not actually used
       max_object_num=100
    )


    '''************************************************************'''
    '''Dataset'''
    # the pretrained model dir (not the backbone)
    pre_model_dir = None
    # evaluate model path, fill the path if you only want to evaluate your model
    evaluate = None
    
    # coco
    # base_path = '/home/jovyan/data-vol-polefs-1/dataset/coco/'
    # train_dataset_path = os.path.join(base_path, 'train2017')
    # val_dataset_path = os.path.join(base_path, 'val2017')
    # dataset_annotations_path = os.path.join(base_path, 'annotations')
    
    # voc
    base_path = '/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/'
    train_dataset_path = os.path.join(base_path, 'train0712')
    val_dataset_path = os.path.join(base_path, 'val07')
    dataset_annotations_path = os.path.join(base_path, 'annotations')
    
    train_dataset = CocoDetection(
        image_root_dir=train_dataset_path,
        annotation_root_dir=dataset_annotations_path,
        set="train0712",
        transform=transforms.Compose([
            RandomColorAndBlur(),
            RandomAffine(prob=0.3),
            RandomFlip(flip_prob=0.5),  # 0.3
            RandomCrop(crop_prob=0.3),
            RandomTranslate(translate_prob=0.3),
            Normalize(),
            # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
            Resize(resize=input_image_size, stride=32, multiscale_range=5)  # multi-scale image
            # Resize(resize=input_image_size)  # uniform-scale image
        ]),
        mosaic=True,
        mosaic_prob=0.5,
        image_size=input_image_size
    )
    val_dataset = CocoDetection(
        image_root_dir=val_dataset_path,
        annotation_root_dir=dataset_annotations_path,
        set="val07",
        transform=transforms.Compose([
            Normalize(),
            Resize(resize=input_image_size),
        ])
    )
