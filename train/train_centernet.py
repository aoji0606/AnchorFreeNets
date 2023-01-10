import sys
import os
import argparse
import random
import shutil
import time
import warnings
import json
from collections import OrderedDict

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")
warnings.filterwarnings('ignore')

import numpy as np
from thop import profile
from thop import clever_format
from tqdm import tqdm
import apex
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from config.config_centernet import Config
from public.dataset.cocodataset import COCODataPrefetcher, Collater, MultiScaleCollater
from public.loss import CenterNetLoss
from public.decoder import CenterNetDecoder
from public import models
from public.utils import get_logger
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch COCO Detection Distributed Training')
    parser.add_argument('--network',
                        type=str,
                        default=Config.network,
                        help='name of network')
    parser.add_argument('--backbone_type',
                        type=str,
                        default=Config.backbone_type,
                        help='name of backbone')
    parser.add_argument('--neck_type',
                        type=str,
                        default=Config.neck_type,
                        help='name of neck')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='use the pretrained backbone')

    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--per_node_batch_size',
                        type=int,
                        default=Config.per_node_batch_size,
                        help='per_node batch size')

    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='num_classes of trained data')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')
    parser.add_argument('--sync_bn',
                        type=bool,
                        default=Config.sync_bn,
                        help='use sync bn or not')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='LOCAL_PROCESS_RANK')


    return parser.parse_args()


def validate(val_dataset, model, decoder, args):
    if args.apex:
        model = model.module
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, decoder, args)

    return all_eval_result


def evaluate_coco(val_dataset, model, decoder, args):
    results, image_ids = [], []
    indexes = []
    device = model.device
    for index in range(len(val_dataset)):
        indexes.append(index)

    batch_size = args.per_node_batch_size
    eval_collater = Collater()
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=eval_collater.next)

    start_time = time.time()

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            images, scales = torch.tensor(data['img']), torch.tensor(data['scale'])
            per_batch_indexes = indexes[i * batch_size:(i + 1) * batch_size]

            images = images.to(device).float()
            cls_heads, reg_heads, center_heads = model(images)
            scores, classes, boxes = decoder(cls_heads, reg_heads, center_heads)

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
                        'image_id':
                        val_dataset.image_ids[index],
                        'category_id':
                        val_dataset.find_category_id_from_coco_label(object_class),
                        'score':
                        object_score,
                        'bbox':
                        object_box,
                    }
                    results.append(image_result)

                image_ids.append(val_dataset.image_ids[index])

                print('{}/{}'.format(index, len(val_dataset)), end='\r')

    testing_time = (time.time() - start_time)
    per_image_testing_time = testing_time / len(val_dataset)

    print(f"per_image_testing_time:{per_image_testing_time:.3f}")

    if not len(results):
        print(f"No target detected in test set images")
        return

    json.dump(results,
              open('{}_{}_{}_{}_bbox_results.json'.format(val_dataset.set_name, Config.network,
                     Config.backbone_type, Config.version), 'w'),
              indent=4)

    # load results in COCO evaluation tool
    coco_true = val_dataset.coco
    coco_pred = coco_true.loadRes('{}_{}_{}_{}_bbox_results.json'.format(val_dataset.set_name, 
                    Config.network, Config.backbone_type, Config.version))

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result


def main():
    args = parse_args()
    global local_rank
    local_rank = args.local_rank
    if local_rank == 0:
        global logger
        logger = get_logger(__name__, args.log)

    torch.cuda.empty_cache()

    #set random seed
    seed = args.local_rank + 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    global gpus_num
    gpus_num = torch.cuda.device_count()
    if local_rank == 0:
        logger.info(f'use {gpus_num} gpus')
        logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    if local_rank == 0:
        logger.info('start loading data')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        Config.train_dataset, shuffle=True)
    collater = Collater()
    # collater = MultiScaleCollater(resize=args.input_image_size,
    #                                         stride=4,)
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.per_node_batch_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=args.num_workers,
                              collate_fn=collater.next,
                              sampler=train_sampler)
    if local_rank == 0:
        logger.info('finish loading data')

    #load model
    model = models.__dict__[args.network](**{
        "pretrained": args.pretrained,
        "backbone_type": args.backbone_type, 
        "neck_type": args.neck_type, 
        "neck_dict": Config.neck_dict, 
        "head_dict": Config.head_dict, 
        "backbone_dict": Config.backbone_dict
    })

    # if Config.pretrained:
        # if args.version == 1 or args.version == 2 or Config.multi_task:
        #     pre_model = torch.load('/home/jovyan/data-vol-polefs-1/yolof_dc5_res50_coco667_withImPre/best.pth', map_location='cpu')
        #     if local_rank == 0:
        #         logger.info(f"pretrained_model: {'/home/jovyan/data-vol-polefs-1/yolof_dc5_res50_coco667_withImPre/best.pth'}")
        # else:
        #     pre_model = torch.load(Config.pre_model_dir, map_location='cpu')
        #     if local_rank == 0:
        #         logger.info(f"pretrained_model: {Config.pre_model_dir}")
    
        # model.load_state_dict(pre_model, strict=False)

    # if Config.multi_task:

    #     # for p in model.backbone.parameters():
    #     #     p.requires_grad = False
    #     # for p in model.clsregcnt_head_1.parameters():
    #     #         p.requires_grad = False

    #log the params and their grad condition
    for name, param in model.named_parameters():
        if local_rank == 0:
            logger.info(f"{name},{param.requires_grad}")

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    if local_rank == 0:
        logger.info(
            f"model: '{args.network}', flops: {flops}, params: {params}")

    criterion = CenterNetLoss(**Config.loss_dict).to(device)
    decoder = CenterNetDecoder(**Config.decoder_dict).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=3,
                                                           verbose=True)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=4, eta_min=1e-6, last_epoch=-1)

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            if local_rank == 0:
                logger.exception(
                    '{} is not a file, please check it again'.format(
                        args.resume))
            sys.exit(-1)
        if local_rank == 0:
            logger.info('start only evaluating')
            logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device('cpu'))
        try:
            checkpoint = {k.lstrip("module."):v for k,v in checkpoint["model_state_dict"].items()}
        except:
            pass
        model.load_state_dict(checkpoint, strict=False)

        if local_rank == 0:
            logger.info(f"start eval.")
            all_eval_result = validate(Config.val_dataset, model, decoder,
                                       args)
            logger.info(f"eval done.")
            if all_eval_result is not None:
                logger.info(
                    f"val: epoch: {checkpoint['epoch']:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
                )

        return

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.apex:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
        model = apex.parallel.DistributedDataParallel(model,
                                                      delay_allreduce=True)
        if args.sync_bn:
            model = apex.parallel.convert_syncbn_model(model)
    else:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)

    best_map = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        if local_rank == 0:
            logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if local_rank == 0:
            logger.info(
                f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, best_map: {checkpoint['best_map']}, "
                f"loss: {checkpoint['loss']:3f}, cls_loss: {checkpoint['cls_loss']:2f}, reg_loss: {checkpoint['reg_loss']:2f}, center_ness_loss: {checkpoint['center_ness_loss']:2f}"
            )

    if local_rank == 0:
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)

    if local_rank == 0:
        logger.info('start training')


    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        cls_losses, reg_losses, center_ness_losses, losses = train(
            device, train_loader, model, criterion, optimizer, scheduler, epoch, args)

        if local_rank == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, heatmap_loss: {cls_losses:.2f}, offset_loss: {reg_losses:.2f}, wh_loss: {center_ness_losses:.2f}, loss: {losses:.2f}"
            )

        if local_rank == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map,
                    'cls_loss': cls_losses,
                    'reg_loss': reg_losses,
                    'center_ness_loss': center_ness_losses,
                    'loss': losses,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.checkpoints, 'latest.pth'))
            logger.info(f"success save epoch {epoch} latest model!")


        if epoch % Config.eval_interval == 0 or epoch % 24 == 0 or epoch == args.epochs:
            if local_rank == 0:
                logger.info(f"start eval.")
                all_eval_result = validate(Config.val_dataset, model, decoder,
                                           args)
                logger.info(f"eval done.")
                if all_eval_result is not None:
                    logger.info(
                        f"val: epoch: {epoch:0>5d}, IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.3f}, IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.3f}, IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.3f}, IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.3f}, IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.3f}, IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.3f}, IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.3f}, IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.3f}, IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
                    )
                    if all_eval_result[0] > best_map:
                        torch.save(model.module.state_dict(),
                                   os.path.join(args.checkpoints, "best.pth"))
                        best_map = all_eval_result[0]
        

    if local_rank == 0:
        logger.info(f"finish training, best_map: {best_map:.3f}")
    training_time = (time.time() - start_time) / 3600
    if local_rank == 0:
        logger.info(
            f"finish training, total training time: {training_time:.2f} hours")
    


def train(device, train_loader, model, criterion, optimizer, scheduler, epoch, args):
    heatmap_losses, offset_losses, wh_losses, losses = [], [], [], []

    torch.cuda.empty_cache()
    # switch to train mode
    model.train()
    optimizer.zero_grad()

    iters = len(train_loader.dataset) // (args.per_node_batch_size * gpus_num)
    # prefetcher = COCODataPrefetcher(train_loader)
    # images, annotations = prefetcher.next()
    iter_index = 1

    one_epoch_start = time.time()
    # while images is not None:
    for one_data in train_loader:
        images, annotations = one_data["img"], one_data["annot"]
        images, annotations = images.to(device).float(), annotations.to(device)
        optimizer.zero_grad()
        heatmap_output, offset_output, wh_output = model(images)
        heatmap_loss, offset_loss, wh_loss = criterion(heatmap_output,
                                                       offset_output,
                                                       wh_output, annotations)

        
        loss = heatmap_loss + offset_loss + wh_loss
        # if heatmap_loss == 0.0 or offset_loss==0.0 or wh_loss == 0.0:
        #     print(f"{local_rank} Zero!!!")
        #     # images, annotations = prefetcher.next()
        #     continue
#*********************************************************************************************************


        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
#         loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        heatmap_losses.append(heatmap_loss.item())
        offset_losses.append(offset_loss.item())
        wh_losses.append(wh_loss.item())
        losses.append(loss.item())
        

        # images, annotations = prefetcher.next()

        if local_rank == 0 and iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>5d}, {iters:0>5d}], heatmap_loss: {heatmap_loss.item():.2f}, offset_loss: {offset_loss.item():.2f}, wh_loss: {wh_loss.item():.2f}, loss_total: {loss.item():.2f}"
            )
            cur_cost = (time.time() - one_epoch_start) / iter_index * (iters * (args.epochs - epoch + 1) - iter_index)
            hour = cur_cost // 3600
            minute = (cur_cost - 3600*hour) // 60
            sec = cur_cost % 60
            day = hour // 24
            hour = hour - day*24
            logger.info(
                f"all training time remain {int(day):0>2d}day {int(hour):0>2d}h {int(minute):0>2d}min {int(sec):0>2d}sec"
            )

        iter_index += 1
    
    one_epoch_cost = time.time() - one_epoch_start
    if local_rank == 0:
        hour = one_epoch_cost // 3600
        minute = (one_epoch_cost - 3600*hour) // 60
        sec = one_epoch_cost % 60
        # day = hour // 24
        # hour = hour - day*24
        logger.info(f"one epoch cost {int(hour):0>2d}h {int(minute):0>2d}min {int(sec):0>2d}sec")

    scheduler.step(np.mean(losses))

    return np.mean(heatmap_losses), np.mean(offset_losses), np.mean(
        wh_losses), np.mean(losses)


if __name__ == '__main__':
    main()
   
