import os
import cv2
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image, ImageEnhance

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

coco_class_colors = [(241, 23, 78), (63, 71, 49),
                     (67, 79, 143), (32, 250, 205), (136, 228, 157),
                     (135, 125, 104), (151, 46, 171), (129, 37, 28),
                     (3, 248, 159), (154, 129, 58), (93, 155, 200),
                     (201, 98, 152), (187, 194, 70), (122, 144, 121),
                     (168, 31, 32), (168, 68, 189), (173, 68, 45),
                     (200, 81, 154), (171, 114, 139), (216, 211, 39),
                     (187, 119, 238), (201, 120, 112), (129, 16, 164),
                     (211, 3, 208), (169, 41, 248), (100, 77, 159),
                     (140, 104, 243), (26, 165, 41), (225, 176, 197),
                     (35, 212, 67), (160, 245, 68), (7, 87, 70), (52, 107, 85),
                     (103, 64, 188), (245, 76, 17), (248, 154, 59),
                     (77, 45, 123), (210, 95, 230), (172, 188, 171),
                     (250, 44, 233), (161, 71, 46), (144, 14, 134),
                     (231, 142, 186), (34, 1, 200), (144, 42, 108),
                     (222, 70, 139), (138, 62, 77),
                     (178, 99, 61), (17, 94, 132), (93, 248, 254),
                     (244, 116, 204), (138, 165, 238), (44, 216, 225),
                     (224, 164, 12), (91, 126, 184), (116, 254, 49),
                     (70, 250, 105), (252, 237, 54), (196, 136, 21),
                     (234, 13, 149), (66, 43, 47), (2, 73, 234), (118, 181, 5),
                     (105, 99, 225), (150, 253, 92), (59, 2, 121),
                     (176, 190, 223), (91, 62, 47), (198, 124, 140),
                     (100, 135, 185), (20, 207, 98), (216, 38, 133),
                     (17, 202, 208), (216, 135, 81), (212, 203, 33),
                     (108, 135, 76), (28, 47, 170), (142, 128, 121),
                     (23, 161, 179), (33, 183, 224)]


class CocoDetection(Dataset):
    def __init__(self,
                 image_root_dir,
                 annotation_root_dir,
                 set='train2017',
                 transform=None,
                 mosaic=False,
                 mosaic_prob=0.5,
                 image_size=512):
        self.image_root_dir = image_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.set_name = set
        self.transform = transform
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.image_size = image_size

        self.coco = COCO(
            os.path.join(self.annotation_root_dir, 'instances_' + self.set_name + '.json')
        )

        self.load_classes()

    def load_classes(self):
        self.image_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(self.cat_ids)
        self.categories.sort(key=lambda x: x['id'])

        # category_id is an original id,coco_id is set from 0 to 79
        self.category_id_to_coco_label = {
            category['id']: i
            for i, category in enumerate(self.categories)
        }
        self.coco_label_to_category_id = {
            v: k
            for k, v in self.category_id_to_coco_label.items()
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # load a mosaic image with target
        if self.mosaic and np.random.rand() < self.mosaic_prob:
            img, annot = self.load_mosaic(idx)
        # load an image with target
        else:
            img = self.load_image(idx)
            annot = self.load_annotations(idx)

        sample = {'img': img, 'annot': annot, 'scale': 1.}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_mosaic(self, index):
        id1 = index
        # random sample other indexs
        ids_list_ = [i for i in range(index)] + [i for i in range(index + 1, len(self.image_ids))]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        # load image and target
        img_lists = [self.load_image(id_) for id_ in ids]
        tg_lists = [self.load_annotations(id_) for id_ in ids]

        mosaic_img = np.zeros([self.image_size[0] * 2, self.image_size[1] * 2, img_lists[0].shape[2]], dtype=np.uint8)

        # mosaic center
        yc = int(random.uniform(self.image_size[0] // 2, 2 * self.image_size[0] - self.image_size[0] // 2))
        xc = int(random.uniform(self.image_size[1] // 2, 2 * self.image_size[1] - self.image_size[1] // 2))

        mosaic_bboxes, mosaic_labels = list(), list()
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            bboxes_i = target_i[:, :4]
            labels_i = target_i[:, 4]

            h0, w0, _ = img_i.shape

            # resize
            if np.random.randint(2):
                # keep aspect ratio
                r = max(self.image_size[0], self.image_size[1]) / max(h0, w0)
                if r != 1:
                    img_i = cv2.resize(img_i, (int(w0 * r), int(h0 * r)))
            else:
                img_i = cv2.resize(img_i, (int(self.image_size[0]), int(self.image_size[1])))
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.image_size[1] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.image_size[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.image_size[1] * 2), min(self.image_size[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            bboxes_i_ = bboxes_i.copy()
            if len(bboxes_i) > 0:
                # a valid target, and modify it.
                bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / w0 + padw)
                bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / h0 + padh)
                bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / w0 + padw)
                bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / h0 + padh)

                mosaic_bboxes.append(bboxes_i_)
                mosaic_labels.append(labels_i)

        # check target
        valid_bboxes, valid_labels = list(), list()
        if len(mosaic_bboxes) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes)
            mosaic_labels = np.concatenate(mosaic_labels)
            # Cutout/Clip targets
            mosaic_bboxes[:, [0, 2]] = np.clip(mosaic_bboxes[:, [0, 2]], 0, 2 * self.image_size[1] - 1)
            mosaic_bboxes[:, [1, 3]] = np.clip(mosaic_bboxes[:, [1, 3]], 0, 2 * self.image_size[0] - 1)
            # x1 = mosaic_bboxes[:, 0]
            # y1 = mosaic_bboxes[:, 1]
            # x2 = mosaic_bboxes[:, 2]
            # y2 = mosaic_bboxes[:, 3]
            # x1 = np.clip(x1, 0, 2 * self.image_size[1], out=x1).reshape(-1, 1)
            # y1 = np.clip(y1, 0, 2 * self.image_size[0], out=y1).reshape(-1, 1)
            # x2 = np.clip(x2, 0, 2 * self.image_size[1], out=x2).reshape(-1, 1)
            # y2 = np.clip(y2, 0, 2 * self.image_size[0], out=y2).reshape(-1, 1)
            # mosaic_bboxes = np.concatenate([x1, y1, x2, y2], axis=1)

            # check boxes
            for box, label in zip(mosaic_bboxes, mosaic_labels):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if bw > 10. and bh > 10.:
                    valid_bboxes.append([x1, y1, x2, y2])
                    valid_labels.append(label)

        # guard against no boxes via resizing
        valid_bboxes = np.array(valid_bboxes).reshape(-1, 4)
        valid_labels = np.array(valid_labels).reshape(-1, 1)
        mosaic_annotations = np.concatenate((valid_bboxes, valid_labels), axis=1)

        return mosaic_img, mosaic_annotations

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_root_dir, image_info['file_name'])
        img = cv2.imread(path)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(path)
        return img.astype(np.float32)

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for _, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            if a['bbox'][2] > 0 and a['bbox'][3] > 0:
                annotation[0, :4] = a['bbox']
                annotation[0, 4] = self.find_coco_label_from_category_id(a['category_id'])

                annotations = np.append(annotations, annotation, axis=0)

        # transform from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def find_coco_label_from_category_id(self, category_id):
        try:
            return self.category_id_to_coco_label[category_id]
        except:
            print(" asdfa ", category_id, "   ", self.coco_label_to_category_id)
            raise ValueError("asdf")

    def find_category_id_from_coco_label(self, coco_label):
        return self.coco_label_to_category_id[coco_label]

    def num_classes(self):
        return 80

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])


class COCODataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_annot = sample['img'], sample['annot']
        except StopIteration:
            self.next_input = None
            self.next_annot = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_annot = self.next_annot.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_input
        annot = self.next_annot
        self.preload()
        return image, annot


class Collater():
    def __init__(self):
        pass

    def next(self, data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]

        imgs = torch.from_numpy(np.stack(imgs, axis=0))

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:

            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * (-1)

            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * (-1)

        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class MultiScaleCollater():
    def __init__(self,
                 resize=512,
                 multi_scale_range=[0.5, 1.5],
                 stride=32,
                 use_multi_scale=False):
        self.resize = resize
        self.multi_scale_range = multi_scale_range
        self.stride = stride
        self.use_multi_scale = use_multi_scale

    def next(self, data):
        if self.use_multi_scale:
            min_resize = int(
                ((self.resize + self.stride) * self.multi_scale_range[0]) //
                self.stride * self.stride)
            max_resize = int(
                ((self.resize + self.stride) * self.multi_scale_range[1]) //
                self.stride * self.stride)

            final_resize = random.choice(
                range(min_resize, max_resize, self.stride))
        else:
            final_resize = self.resize

        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]

        padded_img = torch.zeros((len(imgs), final_resize, final_resize, 3))

        for i, image in enumerate(imgs):
            height, width, _ = image.shape
            max_image_size = max(height, width)
            resize_factor = final_resize / max_image_size
            resize_height, resize_width = int(height * resize_factor), int(
                width * resize_factor)

            image = cv2.resize(image, (resize_width, resize_height))
            padded_img[i, 0:resize_height,
            0:resize_width] = torch.from_numpy(image)

            annots[i][:, :4] *= resize_factor
            scales[i] = scales[i] * resize_factor

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:

            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * (-1)

            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[
                        idx, :annot.shape[0], :] = torch.from_numpy(annot)
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * (-1)

        padded_img = padded_img.permute(0, 3, 1, 2).contiguous()

        return {'img': padded_img, 'annot': annot_padded, 'scale': scales}


class RandomAffine():
    def __init__(self, prob=0.3, ang=10.0, zoom=1.0):
        self.prob = prob
        self.angle = ang
        self.zoom = zoom

    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.prob:
            image, annots, scale = sample['img'], sample['annot'], sample['scale']
            angle = np.random.randint(-self.angle, self.angle)
            rot_image, rot_annots = self.Rotate(image, annots, angle, self.zoom)

            return {'img': rot_image, 'annot': rot_annots, 'scale': scale}

        return sample

    def Rotate(self, img, annos, angle, zoom):
        h, w, _ = img.shape
        rangle = np.deg2rad(angle)
        nw = int(h * abs(np.sin(rangle)) + w * abs(np.cos(rangle)))
        nh = int(w * abs(np.sin(rangle)) + h * abs(np.cos(rangle)))
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, zoom)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        for ann in annos:
            xmin = ann[0]
            ymin = ann[1]
            xmax = ann[2]
            ymax = ann[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))
            # point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
            # point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
            # point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            ann[:4] = np.array([rx_min, ry_min, rx_max, ry_max])

        return rot_img, annos


class RandomFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if np.random.uniform(0, 1) < self.flip_prob:
            image = image[:, ::-1, :]

            _, width, _ = image.shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            annots[:, 0] = width - x2
            annots[:, 2] = width - x1
            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class RandomCrop(object):
    def __init__(self, crop_prob=0.5):
        self.crop_prob = crop_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.crop_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ], axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_left_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_up_trans)))
            crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_right_trans)))
            crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class RandomTranslate(object):
    def __init__(self, translate_prob=0.5):
        self.translate_prob = translate_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.translate_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = random.uniform(-(max_left_trans - 1), (max_right_trans - 1))
            ty = random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class Normalize(object):
    def __init__(self, mean=None, std=None, is_rgb=False):
        self.mean = mean
        self.std = std
        self.is_rgb = is_rgb

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        # ********************************************************************************************************************
        if not self.is_rgb:
            image = image / 255

        if self.mean and self.std:
            image[:, :, 0] = (image[:, :, 0] - self.mean[0]) / self.std[0]
            image[:, :, 1] = (image[:, :, 1] - self.mean[1]) / self.std[1]
            image[:, :, 2] = (image[:, :, 2] - self.mean[2]) / self.std[2]
        # ****************************************************************************************************************************

        sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class Resize(object):
    def __init__(self, resize=(512, 512), stride=32, multiscale_range=0):
        """
        random_scale e.g. (0.5, 1)
        """
        self.resize = resize
        self.stride = stride
        self.multiscale_range = multiscale_range

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        height, width, _ = image.shape

        # resize image
        # max_image_size = max(height, width)

        _resize = max(self.resize[0], self.resize[1])
        if self.multiscale_range > 0:
            min_resize = int(((_resize + self.stride) // self.stride - self.multiscale_range) * self.stride)
            max_resize = int(
                ((_resize + self.stride) // self.stride + self.multiscale_range) * self.stride)  # int(_resize) #
            _resize = random.choice(range(min_resize, max_resize, self.stride))

        # resize_factor = _resize / max_image_size
        resize_factor = min(1. * self.resize[0] / height, 1. * self.resize[1] / width)
        resize_height, resize_width = int(height * resize_factor), int(width * resize_factor)
        image = cv2.resize(image, (resize_width, resize_height))

        copy_height = min(resize_height, self.resize[0])
        copy_width = min(resize_width, self.resize[1])

        # copy to new image
        new_image = np.zeros((self.resize[0], self.resize[1], 3))
        new_image[:copy_height, :copy_width] = image[:copy_height, :copy_width]

        # alter anno and scale
        # annots[:, [0, 1]] = max(annots[:, [0, 1]] * resize_factor, 0)
        # annots[:, [2]] = min(annots[:, [2]] * resize_factor, copy_width)
        # annots[:, [3]] = min(annots[:, [3]] * resize_factor, copy_width)
        annots[:, :4] = annots[:, :4] * resize_factor
        annots[annots[:, 0] <= 0, 0] = 0
        annots[annots[:, 1] <= 0, 1] = 0
        annots[annots[:, 2] >= copy_width, 2] = copy_width
        annots[annots[:, 3] >= copy_height, 3] = copy_height

        scale = scale * resize_factor

        return {
            'img': torch.from_numpy(new_image),
            'annot': torch.from_numpy(annots),
            'scale': scale
        }


class RandomColorAndBlur(object):
    def __init__(self, brightness_factor=0.3, contrast_factor=0.3, saturation_factor=0.3, hue_factor=0.3,
                 blur_vari=0.3):
        """
        @description  : random color and GaussianBlur transform
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.blur_vari = blur_vari

    def adjust_brightness(self, img, brightness_factor):
        """Adjust brightness of an Image.
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.
        Returns:
            numpy ndarray: Brightness adjusted image.
        """
        if np.random.uniform(0, 1) < brightness_factor:
            brightness_factor = 1 + np.random.uniform(-brightness_factor, brightness_factor)
        else:
            return img

        table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
        # same thing but a bit slower
        # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if img.shape[2] == 1:
            return cv2.LUT(img, table)[:, :, np.newaxis]
        else:
            return cv2.LUT(img, table)

    def adjust_contrast(self, img, contrast_factor):
        """Adjust contrast of an mage.
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            contrast_factor (float): How much to adjust the contrast. Can be any
                non negative number. 0 gives a solid gray image, 1 gives the
                original image while 2 increases the contrast by a factor of 2.
        Returns:
            numpy ndarray: Contrast adjusted image.
        """
        # much faster to use the LUT construction than anything else I've tried
        # it's because you have to change dtypes multiple times

        # input is RGB
        if np.random.uniform(0, 1) < contrast_factor:
            contrast_factor = 1 + np.random.uniform(-contrast_factor, contrast_factor)
        else:
            return img

        if img.ndim > 2 and img.shape[2] == 3:
            mean_value = round(cv2.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0])
        elif img.ndim == 2:
            # grayscale input
            mean_value = round(cv2.mean(img)[0])
        else:
            # multichannel input
            mean_value = round(np.mean(img))

        table = np.array([(i - mean_value) * contrast_factor + mean_value for i in range(0, 256)]).clip(0, 255).astype(
            'uint8')
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(contrast_factor)
        if img.ndim == 2 or img.shape[2] == 1:
            return cv2.LUT(img, table)[:, :, np.newaxis]
        else:
            return cv2.LUT(img, table)

    def adjust_saturation(self, img, saturation_factor):
        """Adjust color saturation of an image.
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            saturation_factor (float):  How much to adjust the saturation. 0 will
                give a black and white image, 1 will give the original image while
                2 will enhance the saturation by a factor of 2.
        Returns:
            numpy ndarray: Saturation adjusted image.
        """
        # ~10ms slower than PIL!
        if np.random.uniform(0, 1) < saturation_factor:
            saturation_factor = 1 + np.random.uniform(-saturation_factor, saturation_factor)
        else:
            return img

        img = Image.fromarray(img)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return np.array(img)

    def adjust_hue(self, img, hue_factor):
        """Adjust hue of an image.
        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.
        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.
        See `Hue`_ for more details.
        .. _Hue: https://en.wikipedia.org/wiki/Hue
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            hue_factor (float):  How much to shift the hue channel. Should be in
                [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                HSV space in positive and negative direction respectively.
                0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                with complementary colors while 0 gives the original image.
        Returns:
            numpy ndarray: Hue adjusted image.
        """
        # After testing, found that OpenCV calculates the Hue in a call to
        # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

        # This function takes 160ms! should be avoided
        if np.random.uniform(0, 1) < hue_factor:
            hue_factor = np.random.uniform(-hue_factor, hue_factor)
        else:
            return img

        img = Image.fromarray(img)
        input_mode = img.mode
        if input_mode in {'L', '1', 'I', 'F'}:
            return np.array(img)

        h, s, v = img.convert('HSV').split()

        np_h = np.array(h, dtype=np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over='ignore'):
            np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

        img = Image.merge('HSV', (h, s, v)).convert(input_mode)
        return np.array(img)

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        image = image.astype(np.uint8)
        image = self.adjust_brightness(image, self.brightness_factor)
        image = self.adjust_contrast(image, self.contrast_factor)
        image = self.adjust_saturation(image, self.saturation_factor)
        image = self.adjust_hue(image, self.hue_factor)
        if np.random.uniform(0, 1) < self.blur_vari:
            image = cv2.GaussianBlur(image, (5, 5), 0).astype(np.float32)

        sample = {'img': image, 'annot': annots, 'scale': scale}
        return sample


if __name__ == '__main__':
    import torchvision.transforms as transforms

    if not os.path.join("/home/jovyan/fast-data/test/"):
        os.mkdir("/home/jovyan/fast-data/test/")

    size = (400, 600)
#     size=(512,512)
    base_path = '/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/'
    train_dataset = CocoDetection(
        image_root_dir=base_path,
        annotation_root_dir=base_path,
        set="test",
        transform=transforms.Compose([
#             RandomColorAndBlur(),
#             RandomAffine(prob=0.3),
#             RandomFlip(flip_prob=0.5),
#             RandomCrop(crop_prob=0.3),
#             RandomTranslate(translate_prob=1),
            Normalize(),
            # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
#             Resize(resize=size, stride=32, multiscale_range=3)  # multi-scale image
            Resize(resize=size)  # uniform-scale image
        ]),
        mosaic=True,
        mosaic_prob=0.5,
        image_size=size
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    it = iter(train_loader)
    for i in range(10):
        sample = it.next()
        im = sample["img"][0].numpy().astype(np.float32) * 255

        bbox = sample["annot"][0].numpy().astype(np.int32)
        print(im.shape)
        print(bbox)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        for item in bbox:
            cv2.rectangle(im, (item[0], item[1]), (item[2], item[3]), (255, 0, 0), 2)
        cv2.imwrite("/home/jovyan/fast-data/test/{}.jpg".format(i + 1), im)
        print("/home/jovyan/fast-data/test/{}.jpg".format(i + 1))
