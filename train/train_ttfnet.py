import os
import sys
import json
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

from thop import profile
from thop import clever_format

try:
    import apex
    from apex import amp
    # from apex.parallel import convert_syncbn_model
    # from apex.parallel import DistributedDataParallel
except:
    print('- Not found apex.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pycocotools.cocoeval import COCOeval

# load base dir
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.append(BASE_DIR)
sys.path.append("../")

from config.config_ttfnet import Config
from public.dataset.cocodataset import Collater
from public.loss import TTFNetLoss, SSIMLoss
from public.decoder import TTFNetDecoder
from public import models
from public.utils import get_logger, param_groups_lrd
from public.utils import StepLRWithWarmup, CosineAnnealingLRWithWarmup


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
    parser.add_argument('--kd',
                        type=bool,
                        default=Config.kd,
                        help='use kd or not')

    return parser.parse_args()


# class Distiller(nn.Module):
#     def __init__(self, teacher, student, techer_names, student_names):
#         super(Distiller, self).__init__()
#         self.teacher = teacher
#         self.student = student
#         self.techer_names = techer_names
#         self.student_names = student_names
#
#         def regitster_hooks(teacher_module, student_module):
#             def hook_teacher_forward(module, input, output):
#                 self.register_buffer(teacher_module, output)
#
#             def hook_student_forward(module, input, output):
#                 self.register_buffer(student_module, output)
#
#             return hook_teacher_forward, hook_student_forward
#
#         teacher_modules = dict(self.teacher.named_modules())
#         student_modules = dict(self.student.named_modules())
#
#         for i, (teacher_name, student_name) in enumerate(zip(techer_names, student_names)):
#             teacher_module = 'teacher_' + teacher_name.replace('.', '_')
#             student_module = 'student_' + student_name.replace('.', '_')
#
#             self.register_buffer(teacher_module, None)
#             self.register_buffer(student_module, None)
#
#             hook_teacher_forward, hook_student_forward = regitster_hooks(teacher_module, student_module)
#             teacher_modules[teacher_name].register_forward_hook(hook_teacher_forward)
#             student_modules[student_name].register_forward_hook(hook_student_forward)
#
#     def forward(self, images):
#         with torch.no_grad():
#             _, _ = self.teacher(images)
#         student_heatmap_output, student_hw_output = self.student(images)
#
#         teacher_feats = []
#         student_feats = []
#         buffer_dict = dict(self.named_buffers())
#         for i, (teacher_name, student_name) in enumerate(zip(self.techer_names, self.student_names)):
#             student_module = 'student_' + teacher_name.replace('.', '_')
#             teacher_module = 'teacher_' + student_name.replace('.', '_')
#
#             teacher_feat = buffer_dict[teacher_module]
#             student_feat = buffer_dict[student_module]
#             teacher_feats.append(teacher_feat)
#             student_feats.append(student_feat)
#
#         return teacher_feats, student_feats, student_heatmap_output, student_hw_output


class Distiller():
    def __init__(self, teacher, student, teacher_kd_layers, student_kd_layers):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_kd_layers = teacher_kd_layers
        self.student_kd_layers = student_kd_layers

        teacher_kd_layer_module = {}
        student_kd_layer_module = {}
        teacher_kd_id_layer = {}
        student_kd_id_layer = {}
        self.teacher_kd_layer_feature = {}
        self.student_kd_layer_feature = {}

        def teacher_farward_hook(module, input, output):
            self.teacher_kd_layer_feature[teacher_kd_id_layer[id(module)]] = output

        def student_farward_hook(module, input, output):
            self.student_kd_layer_feature[student_kd_id_layer[id(module)]] = output

        for layer, module in self.teacher.named_modules():
            if layer in self.teacher_kd_layers:
                module.register_forward_hook(teacher_farward_hook)
                teacher_kd_layer_module[layer] = module
                teacher_kd_id_layer[id(module)] = layer

        for layer, module in self.student.named_modules():
            if layer in self.student_kd_layers:
                module.register_forward_hook(student_farward_hook)
                student_kd_layer_module[layer] = module
                student_kd_id_layer[id(module)] = layer

    def forward(self, images):
        self.teacher_kd_layer_feature = {}
        self.student_kd_layer_feature = {}

        with torch.no_grad():
            _, _ = self.teacher(images)
        student_heatmap_output, student_hw_output = self.student(images)

        teacher_feats = []
        student_feats = []
        for teacher_layer, student_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
            teacher_feats.append(self.teacher_kd_layer_feature[teacher_layer])
            student_feats.append(self.student_kd_layer_feature[student_layer])

        return teacher_feats, student_feats, student_heatmap_output, student_hw_output


total_mAP, total_mAP50 = [], []
total_heatmap_losses, total_hw_losses, total_kd_losses, total_losses = [], [], [], []


def train(device, train_loader, student, distiller, criterion, kd_criterion, optimizer, scheduler, epoch, args):
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    student.train()

    iter_index = 1
    iters = len(train_loader.dataset) // (args.per_node_batch_size * gpus_num)
    heatmap_losses, hw_losses, kd_losses, losses = [], [], [], []
    one_epoch_start = time.time()

    for one_data in train_loader:
        images, annotations = one_data["img"], one_data["annot"]
        images, annotations = images.to(device).float(), annotations.to(device)

        if args.kd:
            teacher_feats, student_feats, student_heatmap_output, student_hw_output = distiller.forward(images)

            kd_loss = 0
            for teacher_feat, student_feat in zip(teacher_feats, student_feats):
                kd_loss += kd_criterion(teacher_feat, student_feat)
            heatmap_loss, hw_loss = criterion(student_heatmap_output, student_hw_output, annotations)
            hard_loss = heatmap_loss + hw_loss
            loss = hard_loss + kd_loss
        else:
            student_heatmap_output, student_hw_output = student(images)
            heatmap_loss, hw_loss = criterion(student_heatmap_output, student_hw_output, annotations)
            kd_loss = torch.tensor([0])
            loss = heatmap_loss + hw_loss

        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(student.parameters(), 35, norm_type=2)
        optimizer.step()

        heatmap_losses.append(heatmap_loss.item())
        hw_losses.append(hw_loss.item())
        kd_losses.append(kd_loss.item())
        losses.append(loss.item())

        if local_rank == 0 and iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>5d}, {iters:0>5d}], "
                f"lr:{optimizer.param_groups[0]['lr']:.6f}, heatmap_loss: {heatmap_loss.item():.2f}, "
                f"hw_loss: {hw_loss.item():.2f}, kd_loss: {kd_loss.item():.2f}, loss_total: {loss.item():.2f}"
            )
            cur_cost = (time.time() - one_epoch_start) / iter_index * (iters * (args.epochs - epoch + 1) - iter_index)
            hour = cur_cost // 3600
            minute = (cur_cost - 3600 * hour) // 60
            sec = cur_cost % 60
            day = hour // 24
            hour = hour - day * 24
            logger.info(
                f"all training time remain {int(day):0>2d}day "
                f"{int(hour):0>2d}h {int(minute):0>2d}min {int(sec):0>2d}sec"
            )

        scheduler.step(epoch=epoch + iter_index / iters - 1)  # step by mini-batch
        iter_index += 1

    one_epoch_cost = time.time() - one_epoch_start
    if local_rank == 0:
        hour = one_epoch_cost // 3600
        minute = (one_epoch_cost - 3600 * hour) // 60
        sec = one_epoch_cost % 60
        # day = hour // 24
        # hour = hour - day*24
        logger.info(f"one epoch cost {int(hour):0>2d}h {int(minute):0>2d}min {int(sec):0>2d}sec")

    # scheduler.step(np.mean(losses))  # for ReduceLROnPlateau scheduler
    # scheduler.step()  # step by epoch

    global total_heatmap_losses, total_hw_losses, total_kd_losses, total_losses
    total_heatmap_losses += heatmap_losses
    total_hw_losses += hw_losses
    total_kd_losses += kd_losses
    total_losses += losses

    return np.mean(heatmap_losses), np.mean(hw_losses), np.mean(kd_losses), np.mean(losses)


@torch.no_grad()
def validate(val_dataset, model, decoder, args):
    if args.apex:
        model = model.module
    # switch to evaluate mode
    model.eval()

    all_eval_result = evaluate_coco(val_dataset, model, decoder, args)

    return all_eval_result


@torch.no_grad()
def evaluate_coco(val_dataset, model, decoder, args):
    results, image_ids = [], []
    indexes = []
    device = torch.device('cuda:0')
    for index in range(len(val_dataset)):
        indexes.append(index)

    batch_size = args.per_node_batch_size
    eval_collater = Collater()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_collater.next
    )

    start_time = time.time()
    for i, data in tqdm(enumerate(val_loader)):
        images, scales = torch.tensor(data['img']), torch.tensor(data['scale'])
        per_batch_indexes = indexes[i * batch_size:(i + 1) * batch_size]

        images = images.to(device).float()
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
    logger.info(
        f"testing_time: {testing_time:.4f}, per_image_testing_time: {per_image_testing_time:.4f}"
    )

    if not len(results):
        print(f"No target detected in test set images")
        return

    json_name = os.path.join(args.checkpoints, "results.json")
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


def main():
    global total_mAP, total_mAP50
    global total_heatmap_losses, total_hw_losses, total_kd_losses, total_losses

    args = parse_args()
    global local_rank
    local_rank = args.local_rank
    if local_rank == 0:
        global logger
        logger = get_logger(__name__, args.log)

        print('*' * 20)
        print("input_image_size", Config.input_image_size[0], Config.input_image_size[1])
        print("per_node_batch_size", Config.per_node_batch_size)
        print("epochs", Config.epochs)
        print("neck_dict", Config.neck_dict)
        print("head_dict", Config.head_dict)
        print("mosaic", Config.mosaic)
        print("kd", Config.kd)
        print('*' * 20)

    torch.cuda.empty_cache()

    # set random seed
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
    train_loader = DataLoader(
        Config.train_dataset,
        batch_size=args.per_node_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collater.next,
        sampler=train_sampler
    )
    if local_rank == 0:
        logger.info('finish loading data')

    # load model
    if args.kd:
        teacher = torch.load(Config.teacher_path)
        kd_criterion = SSIMLoss().to(device)
    else:
        teacher = None
        kd_criterion = None

    student = models.__dict__[args.network](**{
        "pretrained": args.pretrained,
        "backbone_type": args.backbone_type,
        "neck_type": args.neck_type,
        "neck_dict": Config.neck_dict,
        "head_dict": Config.head_dict,
        "backbone_dict": Config.backbone_dict
    })

    flops_input = torch.randn(1, 3, args.input_image_size[0], args.input_image_size[1])
    if args.kd:
        teacher_flops, teacher_params = profile(teacher, inputs=(flops_input,), verbose=False)
        teacher_flops, teacher_params = clever_format([teacher_flops, teacher_params], "%.2f")
    student_flops, student_params = profile(student, inputs=(flops_input,), verbose=False)
    student_flops, student_params = clever_format([student_flops, student_params], "%.2f")
    if local_rank == 0:
        if args.kd:
            logger.info(f"teacher, flops: {teacher_flops}, params: {teacher_params}")
        logger.info(f"student, flops: {student_flops}, params: {student_params}")

    if args.kd:
        teacher = teacher.to(device)
        teacher.eval()
    student = student.to(device)
    criterion = TTFNetLoss(**Config.loss_dict).to(device)  # CenterNetLoss(**Config.loss_dict).to(device)
    decoder = TTFNetDecoder(**Config.decoder_dict).to(device)  # CenterNetDecoder(**Config.decoder_dict).to(device)

    # for SGD optimizer
    # param_groups, param_groups_names = param_groups_lrd(model, lr=args.lr, weight_decay=0.0004)
    # if local_rank == 0:
    #     print('parameter groups: \n{}'.format(json.dumps(param_groups_names, indent=2)))
    # optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    # for AdamW optimizer
    param_groups, param_groups_names = param_groups_lrd(student, lr=args.lr, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = StepLRWithWarmup(
        optimizer, warmup_epochs=Config.warmup_epochs, warmup_ratio=Config.warmup_ratio,
        milestones=Config.milestones, gamma=0.1
    )
    # Note: coslr is worse than steplr
    # scheduler = CosineAnnealingLRWithWarmup(
    #     optimizer, total_epochs=Config.epochs, warmup_epochs=Config.warmup_epochs, warmup_ratio=Config.warmup_ratio
    # )

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            if local_rank == 0:
                logger.exception('{} is not a file, please check it again'.format(args.resume))
            sys.exit(-1)

        if local_rank == 0:
            logger.info('start only evaluating')
            logger.info(f"start resuming model from {args.evaluate}")

        checkpoint = torch.load(args.evaluate, map_location=torch.device('cpu'))
        try:
            checkpoint = {k.lstrip("module."): v for k, v in checkpoint["model_state_dict"].items()}
        except:
            pass
        student.load_state_dict(checkpoint, strict=False)

        if local_rank == 0:
            logger.info(f"start eval.")
            all_eval_result = validate(Config.val_dataset, student, decoder, args)
            logger.info(f"eval done.")
            if all_eval_result is not None:
                logger.info(
                    f"val: epoch: {checkpoint['epoch']:0>5d},\n"
                    f"IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.4f},\n"
                    f"IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.4f},\n"
                    f"IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.4f},\n"
                    f"IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.4f},\n"
                    f"IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.4f},\n"
                    f"IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.4f},\n"
                    f"IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.4f},\n"
                    f"IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.4f},\n"
                    f"IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.4f},\n"
                    f"IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.4f},\n"
                    f"IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.4f},\n"
                    f"IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.4f}"
                )
        return

    if args.sync_bn:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)

    if args.apex:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        student, optimizer = amp.initialize(student, optimizer, opt_level='O0')
        student = apex.parallel.DistributedDataParallel(student, delay_allreduce=True)
        if args.sync_bn:
            student = apex.parallel.convert_syncbn_model(student)
    else:
        if args.kd:
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[local_rank], output_device=local_rank)
        student = nn.parallel.DistributedDataParallel(student, device_ids=[local_rank], output_device=local_rank)

    best_map = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        if local_rank == 0:
            logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map = checkpoint['best_map']
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if local_rank == 0:
            logger.info(
                f"finish resuming model from {args.resume}, "
                f"epoch {checkpoint['epoch']}, best_map: {checkpoint['best_map']:.4f}, "
                f"heatmap_loss: {checkpoint['heatmap_loss']:.2f}, hw_loss: {checkpoint['hw_loss']:.2f}, "
                f"kd_loss: {checkpoint['kd_loss']:.2f}, total_loss: {checkpoint['total_loss']:.2f}"
            )

    if args.kd:
        distiller = Distiller(teacher, student, Config.teacher_kd_layers, Config.student_kd_layers)
    else:
        distiller = None

    if local_rank == 0:
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)

    if local_rank == 0:
        logger.info('start training')

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        heatmap_loss, hw_loss, kd_loss, total_loss = train(
            device, train_loader, student, distiller, criterion, kd_criterion, optimizer, scheduler, epoch, args
        )

        if local_rank == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, heatmap_loss: {heatmap_loss:.2f},"
                f"hw_loss: {hw_loss:.2f}, kd_loss: {kd_loss:.2f}, total_loss: {total_loss:.2f}"
            )

        if epoch % Config.eval_interval == 0 or epoch == args.epochs:
            if local_rank == 0:
                logger.info(f"start eval.")
                all_eval_result = validate(Config.val_dataset, student, decoder, args)
                logger.info(f"eval done.")
                if all_eval_result is not None:
                    total_mAP.append(all_eval_result[0])
                    total_mAP50.append(all_eval_result[1])
                    logger.info(
                        f"val: epoch: {epoch:0>5d},\n"
                        f"IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.4f},\n"
                        f"IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.4f},\n"
                        f"IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.4f},\n"
                        f"IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.4f},\n"
                        f"IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.4f},\n"
                        f"IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.4f},\n"
                        f"IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.4f},\n"
                        f"IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.4f},\n"
                        f"IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.4f},\n"
                        f"IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.4f},\n"
                        f"IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.4f},\n"
                        f"IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.4f}"
                    )
                    if all_eval_result[0] > best_map:
                        torch.save(student.module.state_dict(), os.path.join(args.checkpoints, "best.pth"))
                        best_map = all_eval_result[0]

        if local_rank == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map,
                    'heatmap_loss': heatmap_loss,
                    'hw_loss': hw_loss,
                    'kd_loss': kd_loss,
                    'total_loss': total_loss,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.checkpoints, 'latest.pth'))
            logger.info(f"success save epoch {epoch} latest model!")

    if local_rank == 0:
        logger.info(f"finish training, best_map: {best_map:.4f}")

    training_time = (time.time() - start_time) / 3600
    if local_rank == 0:
        total_mAP = np.array(total_mAP)
        total_mAP50 = np.array(total_mAP50)
        total_heatmap_losses = np.array(total_heatmap_losses)
        total_hw_losses = np.array(total_hw_losses)
        total_kd_losses = np.array(total_kd_losses)
        total_losses = np.array(total_losses)
        np.save(os.path.join(args.checkpoints, "total_mAP.npy"), total_mAP)
        np.save(os.path.join(args.checkpoints, "total_mAP50.npy"), total_mAP50)
        np.save(os.path.join(args.checkpoints, "total_heatmap_losses.npy"), total_heatmap_losses)
        np.save(os.path.join(args.checkpoints, "total_hw_losses.npy"), total_hw_losses)
        np.save(os.path.join(args.checkpoints, "total_kd_losses.npy"), total_kd_losses)
        np.save(os.path.join(args.checkpoints, "total_losses.npy"), total_losses)

        logger.info(f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    main()
