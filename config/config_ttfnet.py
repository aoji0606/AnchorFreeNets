import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")

from public.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize, \
    RandomAffine, RandomColorAndBlur

import torchvision.transforms as transforms


class Config(object):
    task_name = "ttfnet"

    # the type of network: FCOS, CenterNet.... in public.models
    network = "TTFNet"

    # the type of backbone: rmobilenet, resnet18, resnet50, swin_t, swin_b, repvgg_a0 ...  in public.backbone
    # backbone_type = "resnet18"
    backbone_type = "mobilenetv2"

    num_classes = 11  # 20 # Note: The number of categories needs to be changed after changing the data set.

    # 2x
    # epochs = 24
    # milestones = [18, 22]

    # 3x
    # epochs = 36
    # milestones = [27, 33]

    # 10x
    epochs = 120
    milestones = [90, 110]

    mosaic = True

    input_image_size = (320, 512)
    per_node_batch_size = 64
    lr = 8e-3 / 20  # adamw lr
    warmup_epochs = 2
    warmup_ratio = 0.2
    num_workers = 8
    print_interval = 10
    eval_interval = 1
    apex = False
    sync_bn = True

    '''************************************************************'''
    '''backbone'''
    # use the pretrained backbone
    pretrained = True
    backbone_dict = dict(out_indices=(3,)) if 'swin' in backbone_type else None

    '''************************************************************'''
    '''neck'''
    # the type of neck: CenterNetNeck, TTFNeck, YolofDC5, RetinaFPN_TransConv, RetinaFPN... in public.neck
    # if you want to use fpn, you must set the neck_dict by yourself!!!!!!!!!!!!!!!
    neck_type = "CenterNetNeck"
    # neck_type = "TTFNeck"

    # use the param when you use normal fpn as neck
    # use the p5 to downsample (set to be True common)
    if "CenterNet" in neck_type:
        neck_out_channles = 64
        # neck_dict = dict(num_layers=3, out_channels=[256, 128, 64], depthwise=False)
        neck_dict = dict(num_layers=3, out_channels=[96, 96, 64], depthwise=False)

    # use the param when you use yolof as neck
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

    # use the param when you use TTF as neck
    if "TTF" in neck_type:
        if "swin" in backbone_type:
            backbone_dict = dict(out_indices=(0, 1, 2, 3))
        if backbone_type == "resnet18":
            neck_out_channles = 64
            neck_dict = dict(out_channels=[256, 128, 64], upsample=False, selayer=False, deformable=False,
                             depthwise=False)
        elif backbone_type == "mobilenetv2":
            neck_out_channles = 24
            neck_dict = dict(out_channels=[96, 32, 24], upsample=False, selayer=False, deformable=False,
                             depthwise=False)

    '''************************************************************'''
    '''Head'''
    head_dict = dict(
        num_classes=num_classes,
        out_channels=neck_out_channles,
        depthwise=True
    )

    '''************************************************************'''
    '''decoder'''
    # down sample strides
    decoder_dict = dict(
        image_w=input_image_size[1],
        image_h=input_image_size[0],
        min_score_threshold=0.01,  # origin: 0.05,
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
        wh_weight=5.,  # 0.1,
        epsilon=1e-4,
        gaussian_radius_alpha=0.54,  # radius for ttfnet
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

    # sdj
    base_path = '/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/'
    # train_annotations_path = os.path.join(base_path, '1107_9w/annotations')
    # val_annotations_path = os.path.join(base_path, '1107_9w/annotations')
    # val_annotations_path = os.path.join(base_path, '1109_1w/annotations')
    # train_annotations_path = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled"
    # val_annotations_path = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled"

    train_annotations_path = base_path
    val_annotations_path = base_path

    train_dataset = CocoDetection(
        image_root_dir=base_path,
        annotation_root_dir=train_annotations_path,
        set="train",
        transform=transforms.Compose([
            RandomColorAndBlur(brightness_factor=0.3, contrast_factor=0.3, saturation_factor=0.3, hue_factor=0.3,
                               blur_vari=0.3),
            RandomAffine(prob=0.3),
            RandomFlip(flip_prob=0.5),
            RandomCrop(crop_prob=0.3),
            RandomTranslate(translate_prob=0.3),
            Normalize(),
            Resize(resize=input_image_size),  # uniform-scale image
            # Actual multiscale ranges: [640 - 3 * 32, 640 + 3 * 32].
            Resize(resize=input_image_size, stride=32, multiscale_range=3)  # multi-scale image
        ]),
        mosaic=mosaic,
        mosaic_prob=0.5,
        image_size=input_image_size
    )

    val_dataset = CocoDetection(
        image_root_dir=base_path,
        annotation_root_dir=val_annotations_path,
        set="test",
        transform=transforms.Compose([
            Normalize(),
            Resize(resize=input_image_size),
        ])
    )

    print("*"*20)
    print("input_image_size", input_image_size[0], input_image_size[1])
    print("per_node_batch_size", per_node_batch_size)
    print("epochs", epochs)
    print("neck_dict", neck_dict)
    print("head_dict", head_dict)
    print("mosaic", mosaic)
    print("*"*20)

    remark = "{}_{}_{}_epoch{}_size{}-{}_mosaic{}".format(
        network, backbone_type, neck_type, epochs, input_image_size[0], input_image_size[1], mosaic)
    # Path to save log
    log = '../logs/{}'.format(remark)
    # Path to store checkpoint model
    checkpoint_path = '../checkpoints/{}'.format(remark)
    # load checkpoint model
    resume = checkpoint_path + '/latest.pth'
