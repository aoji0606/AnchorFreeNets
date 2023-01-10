import copy
import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from argparse import ArgumentParser
from tqdm import tqdm
from shutil import rmtree


def compute_iou(box1, box2):  # (x1,y1,x2,y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

    return intersection / union


def analysis(coco_gt, coco_pred, img_id, cls_map, conf, save, data_dir, out_dir):
    # TP_dir = os.path.join(out_dir, "TP")
    FP_dir = os.path.join(out_dir, "FP")
    FN_dir = os.path.join(out_dir, "FN")
    TP_data = {}
    FP_data = {}
    FN_data = {}

    img_info = coco_gt.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    file_name_ = file_name.replace("/", "_")

    # load gt box
    annIds = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(annIds)
    if len(anns) == 0:  # no gt box
        return TP_data, FP_data, FN_data
    else:  # convert (x,y,w,h) to (x1,y1,x2,y2)
        gt_boxes, gt_labels = [], []
        for item in anns:
            gt_labels.append(item["category_id"])
            gt_boxes.append(item["bbox"])

        gt_labels = np.array(gt_labels)
        gt_boxes = np.array(gt_boxes)
        gt_boxes[:, 2] += gt_boxes[:, 0]
        gt_boxes[:, 3] += gt_boxes[:, 1]

    # load pred box
    detIds = coco_pred.getAnnIds(imgIds=img_id)
    dets = coco_pred.loadAnns(detIds)
    if len(dets) == 0:  # no pred box
        for i in range(len(gt_boxes)):
            cls = gt_labels[i]
            cls_name = cls_map[cls]
            gt_box = gt_boxes[i]
            key = f"{cls_name}_{file_name_}"
            if key not in FN_data.keys():
                FN_data[key] = []
            FN_data[key].append(gt_box)
    else:
        det_boxes, det_labels, det_scores = [], [], []
        for item in dets:
            det_labels.append(item["category_id"])
            det_boxes.append(item["bbox"])
            det_scores.append(item['score'])

        # convert (x,y,w,h) to (x1,y1,x2,y2)
        det_boxes = np.array(det_boxes)
        det_labels = np.array(det_labels)
        det_scores = np.array(det_scores)
        det_boxes[:, 2] += det_boxes[:, 0]
        det_boxes[:, 3] += det_boxes[:, 1]

        # set confidence threshold
        idx = det_scores > conf
        det_scores = det_scores[idx]
        det_boxes = det_boxes[idx]
        det_labels = det_labels[idx]

        # sort by score
        idx = np.argsort(-det_scores)
        det_scores = det_scores[idx]
        det_boxes = det_boxes[idx]
        det_labels = det_labels[idx]

        tp_gt_indexs = []
        tp_det_indexs = []
        for i in range(len(gt_boxes)):
            assign_flag = False

            for j in range(len(det_boxes)):
                if gt_labels[i] == det_labels[j]:
                    cls = gt_labels[i]
                    cls_name = cls_map[cls]
                    gt_box = gt_boxes[i]
                    det_box = det_boxes[j]
                    det_score = det_scores[j]
                    iou = compute_iou(gt_box, det_box)

                    # tp
                    if iou > 0.5 and (not assign_flag):
                        assign_flag = True  # one gt match one det
                        file_name_ = file_name.replace("/", "_")
                        key = f"{cls_name}_{file_name_}"
                        if key not in TP_data.keys():
                            TP_data[key] = []
                        TP_data[key].append([gt_boxes[i], det_boxes[j], det_scores[j]])
                        tp_gt_indexs.append(i)
                        tp_det_indexs.append(j)

        # fn
        for i in range(len(gt_boxes)):
            if i not in tp_gt_indexs:
                cls = gt_labels[i]
                cls_name = cls_map[cls]
                file_name_ = file_name.replace("/", "_")
                key = f"{cls_name}_{file_name_}"
                if key not in FN_data.keys():
                    FN_data[key] = []
                FN_data[key].append(gt_boxes[i])  # miss box, append gt box

        # fp
        for j in range(len(det_boxes)):
            if j not in tp_det_indexs:
                cls = det_labels[j]
                cls_name = cls_map[cls]
                file_name_ = file_name.replace("/", "_")
                key = f"{cls_name}_{file_name_}"
                if key not in FP_data.keys():
                    FP_data[key] = []
                FP_data[key].append([det_boxes[j], det_scores[j]])  # error box, append pred box and score

    # draw box in image
    if save:
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path)

        # draw gt box
        if len(anns):
            for i in range(len(gt_boxes)):
                gt_box = gt_boxes[i]
                label = gt_labels[i]
                name = cls_map[label]
                x1, y1, x2, y2 = gt_box[0], gt_box[1], gt_box[2], gt_box[3]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # green

        # draw pred box
        if len(dets):
            for i in range(len(det_boxes)):
                det_box = det_boxes[i]
                x1, y1, x2, y2 = det_box[0], det_box[1], det_box[2], det_box[3]
                label = det_labels[i]
                name = cls_map[label]
                score = det_scores[i]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # blue
                cv2.putText(img, f'{score:.2f}', (int(x1), int(y1)), 0, 1, (255, 0, 0), 2)  # blue

        # draw tp
        # for key in TP_data.keys():
        #     img_tp = copy.deepcopy(img)
        #     val = TP_data[key]
        #     for v in val:
        #         x1_gt, y1_gt, x2_gt, y2_gt = v[0][0], v[0][1], v[0][2], v[0][3]
        #         x1_dt, y1_dt, x2_dt, y2_dt = v[1][0], v[1][1], v[1][2], v[1][3]
        #         score = v[2]
        #         cv2.rectangle(img_tp, (int(x1_gt), int(y1_gt)), (int(x2_gt), int(y2_gt)), (0, 255, 0), 2)  # green
        #         cv2.rectangle(img_tp, (int(x1_dt), int(y1_dt)), (int(x2_dt), int(y2_dt)), (255, 0, 0), 2)  # blue
        #         cv2.putText(img_tp, f'{score:.2f}', (int(x1_dt), int(y1_dt)), 0, 1, (255, 0, 0), 2)  # blue
        #     cv2.imwrite(os.path.join(TP_dir, key), img_tp)

        # draw fp
        for key in FP_data.keys():
            img_fp = copy.deepcopy(img)
            val = FP_data[key]
            for v in val:
                x1, y1, x2, y2 = v[0][0], v[0][1], v[0][2], v[0][3]
                score = v[1]
                cv2.rectangle(img_fp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # red
                cv2.putText(img_fp, f'{score:.2f}', (int(x1), int(y1)), 0, 1, (0, 0, 255), 2)  # red
            cv2.imwrite(os.path.join(FP_dir, key), img_fp)

        # draw fn
        for key in FN_data.keys():
            img_fn = copy.deepcopy(img)
            val = FN_data[key]
            for v in val:
                x1, y1, x2, y2 = v[0], v[1], v[2], v[3]
                cv2.rectangle(img_fn, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # red
            cv2.imwrite(os.path.join(FN_dir, key), img_fn)

    return TP_data, FP_data, FN_data


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s

    ap = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2]) * 100

    return ap


def analyze_results(pred_file, gt_file, conf, save, data_dir, out_dir):
    if save:
        if os.path.exists(out_dir):
            rmtree(out_dir)

        # os.makedirs(os.path.join(out_dir, "TP"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "FP"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "FN"), exist_ok=True)

    # load gt
    coco_gt = COCO(gt_file)
    cls_ids = coco_gt.getCatIds()
    cls_info = coco_gt.loadCats(cls_ids)
    cls_map = {item['id']: item['name'] for item in cls_info}
    img_ids = coco_gt.getImgIds()

    # load pred
    coco_pred = coco_gt.loadRes(pred_file)

    # coco eval
    aps = []
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    for cls_id in tqdm(cls_ids):
        one_cls_ap = summarize(coco_eval, catId=cls_id)
        aps.append(round(one_cls_ap, 1))

    # precision recall f1
    res = {}
    for catid, catname in cls_map.items():
        res[f'{catname}_TP'] = 0
        res[f'{catname}_FP'] = 0
        res[f'{catname}_FN'] = 0

    for img_id in tqdm(img_ids):
        TP_data, FP_data, FN_data = analysis(coco_gt, coco_pred, img_id, cls_map, conf, save, data_dir, out_dir)

        for key in TP_data.keys():
            val = TP_data[key]
            cls_name = key.split("_")[0]
            res[f'{cls_name}_TP'] += len(val)

        for key in FP_data.keys():
            val = FP_data[key]
            cls_name = key.split("_")[0]
            res[f'{cls_name}_FP'] += len(val)

        for key in FN_data.keys():
            val = FN_data[key]
            cls_name = key.split("_")[0]
            res[f'{cls_name}_FN'] += len(val)

    # print
    keys = list(res.keys())
    errors = []
    print("%12s\t %5s %5s %5s %10s %10s %10s %10s" % ("name", "tp", "fp", "fn", "precision", "recall", "f1", "ap"))
    for i in range(len(cls_map)):
        name = keys[i * 3].split('_')[0]
        try:
            tp = res[keys[i * 3]]
            fp = res[keys[i * 3 + 1]]
            fn = res[keys[i * 3 + 2]]
            precision = 100 * tp / (tp + fp)
            recall = 100 * tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            print("%10s\t %5d %5d %5d %10.2f %10.2f %10.2f %10.1f" % (name, tp, fp, fn, precision, recall, f1, aps[i]))
        except:
            errors.append(name)

    if errors:
        print("error:", *errors)


def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('--pred', default="./test_TTFNet_mobilenetv2_bbox_results.json")
    parser.add_argument('--gt', default='/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/instances_test.json')
    parser.add_argument('--conf', default=0.3)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--data_dir', default="/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/")
    parser.add_argument('--out_dir', default="/home/jovyan/fast-data/sdj_res/")
    args = parser.parse_args()

    analyze_results(args.pred, args.gt, args.conf, args.save, args.data_dir, args.out_dir)


if __name__ == '__main__':
    main()
