# -*- coding=utf-8 -*-
#!/usr/bin/python

"""
This script is used to convert a voc dataset into coco format.
1. download the voc datasets.
2. adapting the voc dataset file dir at (the end of) the script.
3. run this script.

Error help:
pip install tqdm glob
"""

import os
import json
import glob
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET

# init param
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param json_file: 导出json文件的路径
    :return: None
    '''
    # base anno dict
    json_dict = {
        "images":[],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    image_id = 1
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    
    for xml_filename in tqdm(xml_list, desc='deal anno'):
        # read xml file
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        
        # read image info
        image_id += 1
        size = get_and_check(root, 'size', 1)
        json_dict['images'].append({
            'file_name': root.find('filename').text,
            'height': int(get_and_check(size, 'height', 1).text),
            'width': int(get_and_check(size, 'width', 1).text),
            'id':image_id
        })
        
        # deal bbox info
        for obj in get(root, 'object'):
            try:
                category = get_and_check(obj, 'name', 1).text
                if category not in categories: # if there is new class, then add a new one
                    categories[category] = len(categories)  # new id 
                    
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
                ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
                xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                assert(xmax > xmin)
                assert(ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)

                json_dict['annotations'].append({
                    'area': o_width * o_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [xmin, ymin, o_width, o_height],
                    'category_id': category_id,
                    'id': bnd_id,
                    'ignore': 0,
                    'segmentation': [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]],  # 设置分割数据，点的顺序为逆时针方向
                })
                bnd_id += 1
            except Exception as e:
                print(xml_filename, e)

    # write categories to json obj
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    # write to json file
    json_data = json.dumps(json_dict)
    with  open(json_file, 'w') as w:
        w.write(json_data)
    

def voc2coco(
    voc_list_filename=['/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'],
    voc_image_dir=['/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/JPEGImages'],
    voc_anno_dir=['/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/Annotations'],
    coco_image_dir='/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/train0712',
    coco_anno_filename='/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/annotations/instances_train0712.json',
):
    assert isinstance(voc_list_filename, list) and isinstance(voc_image_dir, list) and isinstance(voc_anno_dir, list)
    assert len(voc_list_filename) == len(voc_image_dir) and len(voc_image_dir) == len(voc_anno_dir)
    
    xml_list = []
    for i in range(len(voc_list_filename)):
        # read voc list
        with open(voc_list_filename[i]) as f:
            voc_list = [l.strip() for l in f.readlines()]

        # copy image
        for image_filename in tqdm(glob.glob(os.path.join(voc_image_dir[i], '*.jpg')), desc='copy image'):
            base_filename = os.path.basename(image_filename)
            if base_filename[:-4] in voc_list:  # if image in voc_list
                shutil.copyfile(image_filename, os.path.join(coco_image_dir, base_filename))

        # copy anno
        for xml_filename in glob.glob(os.path.join(voc_anno_dir[i], '*.xml')):
            base_filename = os.path.basename(xml_filename)
            if base_filename[:-4] in voc_list:
                xml_list.append(xml_filename)

    convert(xml_list, coco_anno_filename)


if __name__ == '__main__':
    # generate cocoformat dirs
    root_path = '/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoforma'
    if not os.path.exists(os.path.join(root_path,'annotations')):
        os.makedirs(os.path.join(root_path,'annotations'))
    if not os.path.exists(os.path.join(root_path, 'train0712')):
        os.makedirs(os.path.join(root_path, 'train0712'))
    if not os.path.exists(os.path.join(root_path, 'val07')):
        os.makedirs(os.path.join(root_path, 'val07'))
    
    # train set
    voc2coco(
        voc_list_filename=[
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
        ],
        voc_image_dir=[
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/JPEGImages',
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2012/JPEGImages',
        ],
        voc_anno_dir=[
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/Annotations',
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2012/Annotations',
        ],
        coco_image_dir='/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/train0712',
        coco_anno_filename='/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/annotations/instances_train0712.json',
    )
    # val set
    voc2coco(
        voc_list_filename=[
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        ],
        voc_image_dir=[
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/JPEGImages',
        ],
        voc_anno_dir=[
            '/home/jovyan/data-vol-polefs-1/dataset/VOCdevkit/VOC2007/Annotations',
        ],
        coco_image_dir='/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/val07',
        coco_anno_filename='/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/annotations/instances_val07.json',
    )
