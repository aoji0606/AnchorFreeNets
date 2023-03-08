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
    # the type of network: FCOS, CenterNet.... in public.models
    network = "TTFNet"

    num_classes = 11

    epoch_rate = 10
    epochs = 10 * epoch_rate
    milestones = [8 * epoch_rate, 9 * epoch_rate]

    mosaic = False

    input_image_size = (288, 512)
    per_node_batch_size = 32
    lr = 8e-3 / 20  # adamw lr
    warmup_epochs = min(epochs // 10, 5)
    warmup_ratio = 0.2
    num_workers = 8
    print_interval = 10
    eval_interval = 1
    apex = False
    sync_bn = True

    kd = True
    teacher_path = "../train/teacher.pth"
    teacher_kd_layers = ["module.neck.public_deconv_head.0.relu",
                         "module.neck.public_deconv_head.2.relu",
                         "module.neck.public_deconv_head.4.relu"]
    student_kd_layers = ["module.neck.public_deconv_head.0.relu",
                         "module.neck.public_deconv_head.2.relu",
                         "module.neck.public_deconv_head.4.relu"]

    '''************************************************************'''
    '''backbone'''
    # the type of backbone: rmobilenet, resnet18, resnet50, swin_t, swin_b, repvgg_a0 ...  in public.backbone
    # backbone_type = "convnext"
    # backbone_type = "resnet18"
    backbone_type = "mobilenetv2"

    pretrained = True
    backbone_dict = dict(out_indices=(3,)) if 'swin' in backbone_type else None

    '''************************************************************'''
    '''neck'''
    # the type of neck: CenterNetNeck, TTFNeck, YolofDC5, RetinaFPN_TransConv, RetinaFPN... in public.neck
    # neck_type = "TTFNeck"
    neck_type = "CenterNetNeck"
    # neck_type = "FPN"

    if neck_type == "FPN":
        neck_out_channles = 64
        neck_dict = dict(out_channels=64)

    if neck_type == "CenterNetNeck":
        neck_out_channles = 64
        neck_dict = dict(num_layers=3, out_channels=[256, 128, 64], upsample=True, depthwise=False)

    if neck_type == "YolofDC5":
        if "Yolof" in neck_type:
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

    if neck_type == "TTFNeck":
        if "swin" in backbone_type:
            backbone_dict = dict(out_indices=(0, 1, 2, 3))
        elif "resnet" in backbone_type:
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

    # voc
    # base_path = "/home/jovyan/data-vol-polefs-1/codebase/dataset/voc_coco_format/"
    # train_dataset_path = os.path.join(base_path, "images", "train2017")
    # val_dataset_path = os.path.join(base_path, "images", "test_07")
    # dataset_annotations_path = os.path.join(base_path, "annotations")

    # sdj
    base_path = '/home/jovyan/fast-data/labeled/'
    train_dataset_path = base_path
    val_dataset_path = base_path
    dataset_annotations_path = base_path

    train_dataset = CocoDetection(
        image_root_dir=train_dataset_path,
        annotation_root_dir=dataset_annotations_path,
        set="train",
        transform=transforms.Compose([
            RandomColorAndBlur(brightness_factor=0.3, contrast_factor=0.3, saturation_factor=0.3, hue_factor=0.3,
                               blur_vari=0.3),
            RandomAffine(prob=0.3),
            RandomFlip(flip_prob=0.5),
            RandomCrop(crop_prob=0.3),
            RandomTranslate(translate_prob=0.3),
            Normalize(),
            # Resize(resize=input_image_size),  # uniform-scale image
            # Actual multiscale ranges: [512 - 5 * 32, 512 + 5 * 32].
            Resize(resize=input_image_size, stride=32, multiscale_range=3)  # multi-scale image
        ]),
        mosaic=mosaic,
        mosaic_prob=0.5,
        image_size=input_image_size
    )

    val_dataset = CocoDetection(
        image_root_dir=val_dataset_path,
        annotation_root_dir=dataset_annotations_path,
        set="test",
        transform=transforms.Compose([
            Normalize(),
            Resize(resize=input_image_size),
        ])
    )

    remark = "{}_{}_{}_epoch{}_size{}-{}_batch{}_mosaic{}_kd{}".format(
        network, backbone_type, neck_type, epochs, input_image_size[0], input_image_size[1], per_node_batch_size,
        mosaic, kd)
    # Path to save log
    log = '../logs/{}'.format(remark)
    # Path to store checkpoint model
    checkpoint_path = '../checkpoints/{}'.format(remark)
    # load checkpoint model
    resume = checkpoint_path + '/latest.pth'
