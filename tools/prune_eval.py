from torch.utils.data import DataLoader
import torch
import json
import os
from tqdm import tqdm
import sys
import time
from pycocotools.cocoeval import COCOeval

sys.path.append("../")

from public import models, decoder
from public.dataset.cocodataset import Collater
from config.config_ttfnet import Config

import torch_pruning as tp


def validate(val_dataset, model, device, decoder, args):
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, device, decoder, args)

    return all_eval_result


def evaluate_coco(val_dataset, model, device, decoder, args):
    results, image_ids = [], []
    indexes = []
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
    print(
        f"testing_time: {testing_time:.3f}, per_image_testing_time: {per_image_testing_time:.3f}"
    )

    if not len(results):
        print(f"No target detected in test set images")
        return

    json_name = '{}_{}_{}_bbox_results.json'.format(
        val_dataset.set_name, Config.network, Config.backbone_type
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


def prune(model):
    inputs = torch.randn(1, 3, Config.input_image_size[0], Config.input_image_size[1])
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    for m in model.modules():
        if m._get_name() == "neck" or m._get_name() == "head":
            ignored_layers.append(m)  # DO NOT prune the final classifier!
    #         if m.out_channels == 11 or m.out_channels == 4:
    #             ignored_layers.append(m) # DO NOT prune the final classifier!

    print(ignored_layers)

    # 2. Pruner initialization
    iterative_steps = 10  # You can prune your model to the target sparsity iteratively.
    pruner = tp.pruner.MagnitudePruner(
        model,
        inputs,
        global_pruning=True,  # If False, a uniform sparsity will be assigned to different layers.
        importance=imp,  # importance criterion for parameter selection
        iterative_steps=iterative_steps,  # the number of iterations to achieve target sparsity
        ch_sparsity=0.05,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, inputs)
    for i in range(iterative_steps):
        # 3. the pruner.step will remove some channels from the model with least importance
        pruner.step()

        # 4. Do whatever you like here, such as fintuning
        macs, nparams = tp.utils.count_ops_and_params(model, inputs)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        print('*' * 10)

    return model


if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cpu")
    model_dir = os.path.join(Config.checkpoint_path, "best.pth")

    model = models.__dict__[Config.network](**{
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type,
        "neck_type": Config.neck_type,
        "neck_dict": Config.neck_dict,
        "head_dict": Config.head_dict,
        "backbone_dict": Config.backbone_dict
    })

    decoder_name = Config.network + "Decoder"

    decoder = decoder.__dict__[decoder_name](**Config.decoder_dict)

    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)

    model = prune(model)

    if use_cuda:
        device = torch.device("cuda:0")
        model = model.to(device)
        decoder = decoder.to(device)
    validate(Config.val_dataset, model, device, decoder, Config)
