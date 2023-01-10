import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")

from public.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    task_name = "flag"
    version = 1
    #the type of network: FCOS, CenterNet.... in public.models
    network = "FCOS"
    #the type of backbone: resnet18, resnet50, swin_t, swin_b...  in public.backbone
    '''****************************************************'''
    '''backbone'''
    backbone_type = "swin_t"
    backbone_dict = None
    if "swin" in backbone_type:
        #set the out stage in transformer to avoid the unused param
        backbone_dict = dict(out_indices=(1,2,3))
    pretrained = True

    '''****************************************************'''
    # Path to save log
    log = '../logs/{}/{}/log_v{}'.format(task_name, network+"_"+backbone_type , version)  
    # Path to store checkpoint model
    checkpoint_path = '../checkpoints/{}/{}/v{}'.format(task_name, network+"_"+backbone_type, version)  
    # load checkpoint model
    resume = checkpoint_path + '/latest.pth'


    input_image_size = 512
    num_classes = 15
    epochs = 24
    per_node_batch_size = 2
    lr = 1e-4
    num_workers = 4
    print_interval = 100
    eval_interval = 1
    apex = False
    sync_bn = False

    
    '''************************************************************'''
    '''neck'''
    #the type of neck: RetinaFPN_TransConv, RetinaFPN, YolofDC5... in public.neck
    neck_type = "RetinaFPN"
    #the downsample stride in neck
    strides = [8, 16, 32, 64, 128]
    #the scales param
    scales = [1.0, 1.0, 1.0, 1.0, 1.0]
    #use the param when you use normal fpn as neck
    neck_out_channles = 256
    neck_dict = dict(planes=neck_out_channles,
                    use_p5=True)
    #use the param when you use yolof as neck
    '''if you use the yolof, you must modify the strides and scales in decoder and loss'''
    if "Yolof" in neck_type:
        neck_out_channles = 512
        neck_dict = dict(dila_dict=dict(encoder_channels=neck_out_channles,
                        block_mid_channels=128,
                        num_residual_blocks=4,
                        block_dilations=[4, 8, 12, 16]))
        strides = [16]
        scales = [1.0]
        if "swin" in backbone_type:
            #set the out stage in transformer to avoid the unused param
            backbone_dict = dict(out_indices=(2,3))

    '''************************************************************'''
    '''Head'''
    head_dict = dict(inplanes=neck_out_channles,
                    num_classes=num_classes,
                    num_layers=4,
                    prior=0.01,
                    use_gn=True,
                    cnt_on_reg=True)

    '''************************************************************'''
    '''decoder'''
    #down sample strides
    decoder_dict = dict(image_w=input_image_size,
                        image_h=input_image_size,
                        strides=strides,
                        top_n=1000,
                        min_score_threshold=0.05,
                        nms_threshold=0.6,
                        max_detection_num=100)


    '''************************************************************'''
    '''Loss''' 
    '''FCOSLoss'''  #set the param in different network
    INF = 99999
    #the limit factor
    mi = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
    if "yolof" in neck_type:
        mi = [[-1,512]] #recommand
    loss_dict = dict(strides=strides,
                    mi=mi,
                    alpha=0.25,
                    gamma=2.,
                    reg_weight=2.,
                    epsilon=1e-4,
                    center_sample_radius=1.5,
                    use_center_sample=True)

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
                                      RandomFlip(flip_prob=0.5),
                                      RandomCrop(crop_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
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
