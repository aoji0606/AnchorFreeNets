import os
import sys
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

from public import models


"""
guide: https://spell.ml/blog/pytorch-quantization-X8e7wBAAACIAHPhT
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
https://pytorch.org/docs/master/quantization.html#prototype-fx-graph-mode-quantization

problem: only CPU version.

python tools/repvgg/static_quantization.py \
    --load ./repvgg_a0_deploy_with_bn.pth \
    --save ./repvgg_a0_deploy_w_bn_sq.pth \
    --config config/config_ttfnet.py
"""


def parse_args():
    parser = argparse.ArgumentParser(description='RepVGG Conversion')
    parser.add_argument('--load', metavar='LOAD', help='path to the weights file')
    parser.add_argument('--save', default='repvgg_a0_deploy_w_bn_sq.pt')
    parser.add_argument('--device', default='cuda')
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
    model = models.__dict__[cfg.network](**model_setting)
    # convert to rep-version with bn
    from tools.repvgg.convert import directly_insert_bn_without_init
    directly_insert_bn_without_init(model)

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

@torch.no_grad()
def run_benchmark(model_file, img_loader, device):
    import time
    elapsed = 0
    model = torch.jit.load(model_file).to(device)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, one_data in enumerate(img_loader):
        images, annotations = one_data["img"], one_data["annot"]
        images, annotations = images.to(device).float(), annotations.to(device)
        if i < num_batches:
            start = time.time()
            heatmap_output, wh_output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed


@torch.no_grad()
def evaluate_coco(val_dataset, model, decoder, args, cfg, device):
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
    print(
        f"testing_time: {testing_time:.3f}, per_image_testing_time: {per_image_testing_time:.3f}"
    )

    if not len(results):
        print(f"No target detected in test set images")
        return

    json_name = '{}_{}_{}_{}_quant_bbox_results.json'.format(
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


def main(eval=True):
    args = parse_args()

    # import config
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cfg = config.Config()

    device = torch.device(args.device)

    # load model
    base_model = load_model(cfg, checkpoint_filename=args.load, deploy=True)
    from tools.repvgg.repttfnet_quantized import RepTTFNetWholeQuant
    qat_model = RepTTFNetWholeQuant(repttfnet_model=base_model)
    qat_model.prepare_quant()

    qat_model = qat_model.to(device)

    # qat_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # modules_to_fuse = list()
    # modules_to_fuse += [['backbone.model.stage0.rbr_reparam.conv', 'backbone.model.stage0.rbr_reparam.bn']]
    # modules_to_fuse += [[f'backbone.model.stage1.{i}.rbr_reparam.conv', f'backbone.model.stage1.{i}.rbr_reparam.bn'] for i in range(2)]
    # modules_to_fuse += [[f'backbone.model.stage2.{i}.rbr_reparam.conv', f'backbone.model.stage2.{i}.rbr_reparam.bn'] for i in range(4)]
    # modules_to_fuse += [[f'backbone.model.stage3.{i}.rbr_reparam.conv', f'backbone.model.stage3.{i}.rbr_reparam.bn'] for i in range(14)]
    # modules_to_fuse += [[f'backbone.model.stage4.{i}.rbr_reparam.conv', f'backbone.model.stage4.{i}.rbr_reparam.bn'] for i in range(1)]
    # fused_model = torch.ao.quantization.fuse_modules(qat_model, modules_to_fuse)

    # prepared_fp32_model = torch.quantization.prepare(qat_model)
    model_int8 = torch.quantization.convert(qat_model.eval())

    # traceed_model = torch.jit.trace(model_int8, torch.rand(1, 3, 512, 512), strict=False)
    # torch.jit.save(traceed_model, args.save)
    torch.jit.save(torch.jit.script(model_int8), args.save)

    # load model
    if eval:
        # from torch.utils.data import DataLoader
        from public import decoder
        # from public.dataset.cocodataset import Collater

        # collater = Collater()
        # dataloader = DataLoader(
        #     cfg.val_dataset,
        #     batch_size=args.eval_batch_size,
        #     shuffle=False, pin_memory=True,
        #     num_workers=args.eval_workers,
        #     collate_fn=collater.next
        # )
        # run_benchmark(args.save, dataloader, device=device)
        
        decoder_dict = {'TTFNet': 'TTFNetDecoder'}
        _decoder = decoder.__dict__[decoder_dict[cfg.network]](**cfg.decoder_dict).cuda()
        all_eval_result = evaluate_coco(
            cfg.val_dataset, torch.jit.load(args.save).to(device), _decoder.to(device), args, cfg, device
        )
        # model = torch.jit.load(args.save)
        # model.eval()
        # output = model(torch.rand(1, 3, 512, 512)) # inference
        # print(output)


if __name__ == '__main__':
    main()
