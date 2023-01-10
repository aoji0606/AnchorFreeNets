import os
import json
import cv2 as cv
from tqdm import tqdm

txt_path = "./res.txt"
json_path = "./res.json"

categories = [{"id": 0, "name": "底座", "supercategory": ''},
              {"id": 1, "name": "体重秤", "supercategory": ''},
              {"id": 2, "name": "鞋子", "supercategory": ''},
              {"id": 3, "name": "插线板", "supercategory": ''},
              {"id": 4, "name": "线团线材", "supercategory": ''},
              {"id": 5, "name": "宠物粪便", "supercategory": ''},
              {"id": 6, "name": "袜子毛巾", "supercategory": ''},
              {"id": 7, "name": "玩具", "supercategory": ''},
              {"id": 8, "name": "垃圾桶", "supercategory": ''},
              {"id": 9, "name": "塑料袋", "supercategory": ''},
              {"id": 10, "name": "花盆", "supercategory": ''}]
paths_id = {}
images = []
annotations = []

txt = open(txt_path, 'r')
lines = txt.readlines()
txt.close()

id = 1
for line in tqdm(lines):
    path, labels = line.strip().split(" ", maxsplit=1)

    if path not in paths_id.keys():
        paths_id[path] = id
        id += 1

id = 1
for line in tqdm(lines):
    path, labels = line.strip().split(" ", maxsplit=1)
    labels = labels.split(' ')
    for label in labels:
        x1, y1, x2, y2, cls = label.split(',')
        x1, y1, x2, y2, cls = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)), int(float(cls))
        w = x2 - x1
        h = y2 - y1
        annotations.append({"id": id,
                            "category_id": cls,
                            "iscrowd": 0,
                            "bbox": [x1, y1, w, h],
                            "image_id": paths_id[path],
                            "area": w * h,
                            "segmentation": []})
        id += 1

for path, id in tqdm(paths_id.items()):
    try:
        name = path.split('/')[-1]
        img = cv.imread(path)
        height, width = img.shape[:2]
        images.append({"id": id,
                       "file_name": name,
                       "height": height,
                       "width": width,
                       "license": 0,
                       "date_captured": "",
                       "flickr_url": "",
                       "coco_url": ""})
    except:
        print(name)

coco_j = open(json_path, 'w')
res = {}
res["info"] = ["none"]
res["licenses"] = ["none"]
res["images"] = images
res["annotations"] = annotations
res["categories"] = categories
json.dump(res, coco_j)
