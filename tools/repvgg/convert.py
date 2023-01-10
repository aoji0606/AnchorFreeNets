# -*- coding=utf-8 -*-
#!/usr/bin/python

"""
This script is used to convert repvgg.
# 1. convert, inference mAP
# 2. add bn, inference mAP

python tools/repvgg/convert.py \
    --load /home/jovyan/AnchorFreeNets/checkpoints/ttfnet/TTFNet_repvgg_a0/v1.0.1/best.pth \
    --convert_save ./repvgg_a0_deploy.pth \
    --add_bn_save ./repvgg_a0_deploy_with_bn.pth \
    --config config/config_ttfnet.py
"""

import os
import sys
import copy
import argparse

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)

import torch
from torch import nn
from torch.backends import cudnn

from public import models
from public import decoder

from public.backbone.repvgg_pruning import RepVGGBlock


def parse_args():
    parser = argparse.ArgumentParser(description='RepVGG Conversion')
    parser.add_argument('--load', metavar='LOAD', help='path to the weights file')
    parser.add_argument('--convert_save', default='repvgg_a0_deploy.pth')
    parser.add_argument('--add_bn_save', default='repvgg_a0_deploy_w_bn.pth')
    parser.add_argument('--eval_batch_size', default=32)
    parser.add_argument('--eval_workers', default=2)
    parser.add_argument('--config', metavar='CONFIG', default='config/config_ttfnet.py')
    return parser.parse_args()


def load_model(cfg, checkpoint_filename=None, deploy=False):
    model_setting = {
        "pretrained": False,
        "backbone_type": cfg.backbone_type, 
        "neck_type": cfg.neck_type, 
        "neck_dict": cfg.neck_dict, 
        "head_dict": cfg.head_dict, 
        "backbone_dict": cfg.backbone_dict,
        "deploy": deploy
    }

    # print('model setting: ', model_setting)
    model = models.__dict__[cfg.network](**model_setting)

    # load state dict
    if checkpoint_filename:
        if os.path.isfile(checkpoint_filename):
            print("=> loading checkpoint '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            # state_dict = model.state_dict()
            # for k in state_dict.keys():
            #     if k in ckpt.keys():
            #         print(k)
            model.load_state_dict(ckpt, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_filename))
    
    return model


def repvgg_model_convert(model, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def update_running_mean_var(x, running_mean, running_var, momentum=0.9, is_first_batch=False):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    if is_first_batch:
        running_mean = mean
        running_var = var
    else:
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var
    return running_mean, running_var


#   Record the mean and std like a BN layer but do no normalization
class BNStatistics(nn.Module):
    def __init__(self, num_features):
        super(BNStatistics, self).__init__()
        shape = (1, num_features, 1, 1)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.zeros(shape))
        self.is_first_batch = True

    def forward(self, x):
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        self.running_mean, self.running_var = update_running_mean_var(x, self.running_mean, self.running_var, momentum=0.9, is_first_batch=self.is_first_batch)
        self.is_first_batch = False
        return x


#   This is designed to insert BNStat layer between Conv2d(without bias) and its bias
class BiasAdd(nn.Module):
    def __init__(self, num_features):
        super(BiasAdd, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(num_features))
    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)


def switch_repvggblock_to_bnstat(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            print('switch to BN Statistics: ', n)
            assert hasattr(block, 'rbr_reparam')
            stat = nn.Sequential()
            stat.add_module('conv', nn.Conv2d(
                block.rbr_reparam.in_channels, block.rbr_reparam.out_channels,
                block.rbr_reparam.kernel_size,
                block.rbr_reparam.stride, block.rbr_reparam.padding,
                block.rbr_reparam.dilation,
                block.rbr_reparam.groups, bias=False))  # Note bias=False
            stat.add_module('bnstat', BNStatistics(block.rbr_reparam.out_channels))
            stat.add_module('biasadd', BiasAdd(block.rbr_reparam.out_channels))  # Bias is here
            stat.conv.weight.data = block.rbr_reparam.weight.data
            stat.biasadd.bias.data = block.rbr_reparam.bias.data
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = stat


def switch_bnstat_to_convbn(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            assert hasattr(block, 'rbr_reparam')
            assert hasattr(block.rbr_reparam, 'bnstat')
            print('switch to ConvBN: ', n)
            conv = nn.Conv2d(
                block.rbr_reparam.conv.in_channels, block.rbr_reparam.conv.out_channels,
                block.rbr_reparam.conv.kernel_size,
                block.rbr_reparam.conv.stride, block.rbr_reparam.conv.padding,
                block.rbr_reparam.conv.dilation,
                block.rbr_reparam.conv.groups, bias=False)
            bn = nn.BatchNorm2d(block.rbr_reparam.conv.out_channels)
            bn.running_mean = block.rbr_reparam.bnstat.running_mean.squeeze()  # Initialize the mean and var of BN with the statistics
            bn.running_var = block.rbr_reparam.bnstat.running_var.squeeze()
            std = (bn.running_var + bn.eps).sqrt()
            conv.weight.data = block.rbr_reparam.conv.weight.data
            bn.weight.data = std
            bn.bias.data = block.rbr_reparam.biasadd.bias.data + bn.running_mean  # Initialize gamma = std and beta = bias + mean

            convbn = nn.Sequential()
            convbn.add_module('conv', conv)
            convbn.add_module('bn', bn)
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = convbn


#   Insert a BN after conv3x3 (rbr_reparam). With no reasonable initialization of BN, the model may break down.
#   So you have to load the weights obtained through the BN statistics (please see the function "insert_bn" in this file).
def directly_insert_bn_without_init(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            print('directly insert a BN with no initialization: ', n)
            assert hasattr(block, 'rbr_reparam')
            convbn = nn.Sequential()
            convbn.add_module('conv', nn.Conv2d(block.rbr_reparam.in_channels, block.rbr_reparam.out_channels,
                                              block.rbr_reparam.kernel_size,
                                              block.rbr_reparam.stride, block.rbr_reparam.padding,
                                              block.rbr_reparam.dilation,
                                              block.rbr_reparam.groups, bias=False))  # Note bias=False
            convbn.add_module('bn', nn.BatchNorm2d(block.rbr_reparam.out_channels))
            #   ====================
            convbn.add_module('relu', nn.ReLU())
            # TODO we moved ReLU from "block.nonlinearity" into "rbr_reparam" (nn.Sequential). This makes it more convenient to fuse operators (see RepVGGWholeQuant.fuse_model) using off-the-shelf APIs.
            block.nonlinearity = nn.Identity()
            #==========================
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = convbn


def insert_bn(model, args, cfg):
    import time
    from torch.utils.data import DataLoader
    from public.loss import TTFNetLoss
    from public.dataset.cocodataset import Collater

    switch_repvggblock_to_bnstat(model)

    cudnn.benchmark = True
    
    collater = Collater()
    train_loader = DataLoader(
        cfg.train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False, pin_memory=True,
        num_workers=args.eval_workers,
        collate_fn=collater.next
    )

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    hm_losses = AverageMeter('hm_loss', ':.4e')
    wh_losses = AverageMeter('wh_loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, hm_losses, wh_losses],
        prefix='BN stat: '
    )

    criterion = TTFNetLoss(**cfg.loss_dict).cuda()

    with torch.no_grad():
        end = time.time()
        for i, one_data in enumerate(train_loader):
            images, annotations = one_data["img"], one_data["annot"]
            images, annotations = images.cuda().float(), annotations.cuda()
            heatmap_output, wh_output = model(images)
            heatmap_loss, wh_loss = criterion(heatmap_output, wh_output, annotations)
            loss = heatmap_loss + wh_loss

            losses.update(loss.item(), images.size(0))
            hm_losses.update(heatmap_loss.item(), images.size(0))
            wh_losses.update(wh_loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
    
    switch_bnstat_to_convbn(model)

    torch.save(model.state_dict(), args.add_bn_save)
    return model


@torch.no_grad()
def validate(val_dataset, model, decoder, args, cfg):
    # switch to evaluate mode
    model.eval()
    all_eval_result = evaluate_coco(val_dataset, model, decoder, args, cfg)
    return all_eval_result


@torch.no_grad()
def evaluate_coco(val_dataset, model, decoder, args, cfg):
    import time
    import json
    from tqdm import tqdm

    from pycocotools.cocoeval import COCOeval

    from torch.utils.data import DataLoader
    from public.dataset.cocodataset import Collater

    results, image_ids = [], []
    indexes = []
    for index in range(len(val_dataset)):
        indexes.append(index)

    batch_size = args.eval_batch_size
    eval_collater = Collater()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=eval_collater.next
    )

    start_time = time.time()
    for i, data in tqdm(enumerate(val_loader)):
        images, scales = torch.tensor(data['img']), torch.tensor(data['scale'])
        per_batch_indexes = indexes[i * batch_size:(i + 1) * batch_size]

        images = images.cuda().float()
        cls_heads, center_heads = model(images)
        scores, classes, boxes = decoder(cls_heads, center_heads)

        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        scales = scales.unsqueeze(-1).unsqueeze(-1)
        boxes /= scales

        for per_image_scores, per_image_classes, per_image_boxes, index in zip(
                scores, classes, boxes, per_batch_indexes):
            # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

            for object_score, object_class, object_box in zip(
                    per_image_scores, per_image_classes, per_image_boxes):
                object_score = float(object_score)
                object_class = int(object_class)
                object_box = object_box.tolist()
                if object_class == -1:
                    break

                image_result = {
                    'image_id': val_dataset.image_ids[index],
                    'category_id': val_dataset.find_category_id_from_coco_label(object_class),
                    'score': object_score,
                    'bbox': object_box,
                }
                results.append(image_result)
            image_ids.append(val_dataset.image_ids[index])
            print('{}/{}'.format(index, len(val_dataset)), end='\r')

    testing_time = (time.time() - start_time)
    per_image_testing_time = testing_time / len(val_dataset)
    print(
        f"testing_time: {testing_time:.3f}, per_image_testing_time: {per_image_testing_time:.3f}"
    )

    if not len(results):
        print(f"No target detected in test set images")
        return

    json_name = '{}_{}_{}_{}_convert_bbox_results.json'.format(
        val_dataset.set_name, cfg.network, cfg.backbone_type, cfg.version
    )
    with open(json_name, 'w') as f:
        json.dump(results, f, indent=4)

    # load results in COCO evaluation tool
    coco_true = val_dataset.coco
    coco_pred = coco_true.loadRes(json_name)

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main(eval=True):
    args = parse_args()

    # import config
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cfg = config.Config()

    # load origin model
    model = load_model(cfg, args.load)
    
    # convert model
    repvgg_model_convert(model, save_path=args.convert_save)
    
    if eval:
        # load converted model
        model = load_model(cfg, args.convert_save, deploy=True).cuda()
        decoder_dict = {'TTFNet': 'TTFNetDecoder'}
        _decoder = decoder.__dict__[decoder_dict[cfg.network]](**cfg.decoder_dict).cuda()

        # model inference
        all_eval_result = validate(cfg.val_dataset, model, _decoder, args, cfg)
        if all_eval_result is not None:
            print(
                '========== converted model results ===========\n'
                f"IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.3f},\n"
                f"IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.3f},\n"
                f"IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.3f},\n"
                f"IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.3f},\n"
                f"IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.3f},\n"
                f"IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.3f},\n"
                f"IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.3f},\n"
                f"IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.3f},\n"
                f"IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.3f},\n"
                f"IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.3f},\n"
                f"IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.3f},\n"
                f"IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
            )
    
    # insert bn
    model = load_model(cfg, args.convert_save, deploy=True).cuda()
    insert_bn(model, args, cfg)


if __name__ == '__main__':
    main()
