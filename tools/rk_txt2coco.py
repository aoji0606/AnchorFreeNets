import os
import json
from tqdm import tqdm

name_id = {}
coco_gt = json.load(open("./instances_demo.json", 'r'))
for image in coco_gt["images"]:
    file_name = image["file_name"].split('/')[-1].split('.')[0]
    image_id = image["id"]
    name_id[file_name] = image_id

txt_root = "./res_inference"
txts = os.listdir(txt_root)
res_json = []
for txt in tqdm(txts):
    lines = open(os.path.join(txt_root, txt), 'r').readlines()
    name = txt.split('.')[0]

    image_id = name_id[name]
    temp_all = []
    for line in lines:
        line = line.strip()
        cls_id, conf, bbox = line.split(' ')
        category_id = int(cls_id)
        score = float(conf)
        x1, y1, x2, y2 = bbox.split(',')
        x = int(x1)
        y = int(y1)
        w = int(x2) - x
        h = int(y2) - y
        bbox = [x, y, w, h]

        temp = {}
        temp["image_id"] = image_id
        temp["category_id"] = category_id
        temp["score"] = score
        temp["bbox"] = bbox
        temp_all.append(temp)

    temp_sorted = sorted(temp_all, key=lambda x: x["score"], reverse=True)
    temp_top100 = temp_sorted[:100]
    for item in temp_top100:
        res_json.append(item)

json.dump(res_json, open("./pred.json", 'w'))
