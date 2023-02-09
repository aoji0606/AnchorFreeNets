import torch
import cv2
import numpy as np
import json
import os
import sys
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

sys.path.append("../")

from public import models, decoder
from config.config_ttfnet import Config

# 使用gpu测试
use_gpu = True
# score阈值
threshold = 0.3
# 保存输出图片
if_save_img = True
# 保存输出标注
if_save_annot = False

# 待测模型地址
model_dir = os.path.join(Config.checkpoint_path, "best.pth")
# 输入图片地址
im_dir = '/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/test/'

# 输出图片地址
out_dir = '/home/jovyan/fast-data/ttf_pred'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# label_map
eval_json_dir = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/1109_1w/annotations/instances_default.json"
with open(eval_json_dir, "r") as f:
    eval_j = json.load(f)
    categories = eval_j["categories"]
    label_map = {k: v["name"] for k, v in enumerate(categories)}

print(label_map)

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
resize = max(Config.input_image_size)

device = torch.device("cpu")
if use_gpu:
    device = torch.device("cuda:0")
    decoder = decoder.to(device)
    model = model.to(device)

pre_model = torch.load(model_dir, map_location=torch.device('cpu'))
# if "latest" in model_dir:
#     pre_model = {k.lstrip("module."):v for k,v in pre_model["model_state_dict"].items()}
model.load_state_dict(pre_model, strict=False)
model.eval()

# 创建输出的json dict
out = {}
font = ImageFont.truetype('han.ttc', 15)
im_list = os.listdir(im_dir)

if if_save_annot:
    txt = open("./res.txt", "w")

with torch.no_grad():
    for item in tqdm(im_list[:3]):
        current_dir = os.path.join(im_dir, item)
        if if_save_annot:
            print(current_dir)
            line = current_dir

        img = cv2.imdecode(np.fromfile(current_dir, dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        height, width, _ = img.shape
        max_image_size = max(height, width)
        resize_factor = resize / max_image_size
        resize_height, resize_width = int(height * resize_factor), int(
            width * resize_factor)
        img = cv2.resize(img, (resize_width, resize_height))
        resized_img = np.zeros((resize, resize, 3))
        resized_img[0:resize_height, 0:resize_width] = img

        # save resized img
        if 0:
            temp = resized_img * 255
            temp = temp.astype(np.uint8)
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, "resized_" + item), temp)

        resized_img = torch.tensor(resized_img).permute(2, 0, 1).float().unsqueeze(0)
        resized_img = resized_img.cuda()

        if "CenterNet" == Config.network:
            cls_heads, reg_heads, center_heads = model(resized_img)
            scores, classes, boxes = decoder(cls_heads, reg_heads, center_heads)
        elif "FCOS" == Config.network:
            heatmap_output, offset_output, wh_output = model(resized_img)
            scores, classes, boxes = decoder(heatmap_output, offset_output,
                                             wh_output)
        elif "TTFNet" == Config.network:
            cls_heads, center_heads = model(resized_img)
            scores, classes, boxes = decoder(cls_heads, center_heads)
        else:
            raise ValueError("Config.network must be FCOS or CenterNet or TTFNet")

        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        scores = scores[0].tolist()
        bboxes = (boxes[0] / resize_factor)

        # 读取图片准备画框
        if if_save_img:
            or_im = cv2.imdecode(np.fromfile(current_dir, dtype=np.uint8), -1)
            pil_image = Image.fromarray(cv2.cvtColor(or_im, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

        out[item] = []
        obj_num = 0
        for i, score in enumerate(scores):
            bbox = bboxes[i].tolist()

            if score > threshold:
                obj_num += 1

                dic = {}
                dic["scores"] = score
                dic["class"] = label_map[int(classes[0][i].item())]
                dic["bbox"] = bbox
                out[item].append(dic)

                if if_save_annot:
                    line += " %.2f,%.2f,%.2f,%.2f,%d" % (bbox[0], bbox[1], bbox[2], bbox[3], int(classes[0][i].item()))

                if if_save_img:
                    pos = (int(bbox[0]), int(bbox[1]))
                    text = label_map[int(classes[0][i].item())] + " " + str(score)[:4]
                    color = (255, 255, 0)
                    draw.text(pos, text, font=font, fill=color)
                    draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                                   outline=(0, 0, 255), width=2)

        if if_save_annot and obj_num:
            line += '\n'
            txt.write(line)

        if if_save_img and scores[0] > threshold:
            cv_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, item), cv_img)
