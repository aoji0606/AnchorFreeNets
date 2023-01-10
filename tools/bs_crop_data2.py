import cv2
import json
import numpy as np
import os
from tqdm import tqdm
from  copy import deepcopy
from pycocotools.coco import COCO
import random
import threading

#原始json
my_json_dir = "/home/jovyan/data-vol-polefs-1/dataset/lanmu/annotations/instances_default_v1_and_v1+.json"
#原始已标注图片
my_data_dir = "/home/jovyan/data-vol-polefs-1/dataset/lanmu/or_images"
#被用于插入crop的未标注图片
wait_dir = '/home/jovyan/data-vol-polefs-1/crop_dataset/or_images_v2/'
#图片输出地址
out_dir = "/home/jovyan/data-vol-polefs-1/dataset/lanmu/lanmu_crop_0630_no_boso"
#json输出地址
out_json_dir = "/home/jovyan/data-vol-polefs-1/dataset/lanmu/annotations/lanmu_crop_0630_no_boso.json"


crop_prob = 0.8
bs_prob = 0.
class_num = 41

try:
    os.mkdir(out_dir)
except:
    os.system(f"rm -rf {out_dir}/*")

with open(my_json_dir, "r") as f:
    my_coco_json = json.load(f)
my_coco = COCO(my_json_dir)

im_wait_file_names = os.listdir(wait_dir)

new_data_coco = deepcopy(my_coco_json)
random.shuffle(im_wait_file_names)

im_wait_file_names = im_wait_file_names
per_class = len(im_wait_file_names) / class_num

new_data_coco["images"] = []
new_data_coco["annotations"] = []
categories = [item["name"] for item in my_coco_json["categories"]]



class myThread (threading.Thread):
    def __init__(self, threadID, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.item = counter
    def run(self):
        print ("开始线程：" ,self.threadID)
        crop(self.item)
        print ("退出线程：" , self.threadID)

def crop(item_ls):
    for item in item_ls:
        im_dirs = im_wait_file_names[int(per_class*(item-1)): int(per_class*item)]
        anns_ids = my_coco.getAnnIds(catIds=item)
        anns = my_coco.loadAnns(ids=anns_ids)
        for idx in range(int(per_class*item)-int(per_class*(item-1))):
            anns_idx = random.randint(0, len(anns_ids)-1)
            im_info = my_coco.loadImgs(ids=anns[anns_idx]["image_id"])

            temp_im_info = deepcopy(im_info[0])
            temp_anns = deepcopy(anns[anns_idx])

            im_or = cv2.imread(os.path.join(wait_dir, im_dirs[idx]))
            try:
                shape = im_or.shape
            except:
                continue
            
            if random.uniform(0,1) < crop_prob:
                
                im_mine = cv2.imread(os.path.join(my_data_dir, im_info[0]["file_name"]))
                h_or, w_or = im_or.shape[:2]
                h_mine, w_mine = im_mine.shape[:2]

                ratio = max(w_or / w_mine, h_or / h_mine)
    #             ratio = min(w_or, h_or) / w_mine
                x1_m, y1_m, w_m, h_m = np.clip(np.array(anns[anns_idx]["bbox"]),0,10000).tolist()
                x = int(x1_m*ratio)
                y = int(y1_m*ratio)
                w = int(w_m*ratio)
                h = int(h_m*ratio)

                #*******************************************
                try:
                    x = random.randint(0, w_or-w)
                    y = random.randint(0, h_or-h)
                except:
                    continue
                #*******************************************

                crop = im_mine[int(y1_m):int(y1_m+h_m), int(x1_m):int(x1_m+w_m)]

                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

                if random.uniform(0,1) < bs_prob:
                    mask = 255 * np.ones(crop.shape, crop.dtype)
                    center = (int(x+w/2), int(y+h/2))
                    im_or = cv2.seamlessClone(crop, im_or, mask, center, cv2.NORMAL_CLONE)
                    cur_name = categories[item-1]+"_boso_"+str(idx)+".jpg"
                else:
                    im_or[y:y+h, x:x+w] = crop
                    cur_name = categories[item-1]+str(idx)+".jpg"

                path = os.path.join(out_dir, cur_name)

                temp_anns["bbox"] = [int(x+w/2)-w/2, int(y+h/2)-h/2, w, h]
                temp_anns["area"] = w*h

                cv2.imwrite(path, im_or)


                temp_im_info["file_name"] = cur_name
                temp_im_info["id"] = idx + 1 + int((item-1)*per_class)
                temp_im_info["height"] = im_or.shape[0]
                temp_im_info["width"] = im_or.shape[1]
                temp_anns["image_id"] = idx + 1 + int((item-1)*per_class)
                temp_anns["id"] = idx + 1 + int((item-1)*per_class)
                new_data_coco["images"].append(temp_im_info)
                new_data_coco["annotations"].append(temp_anns)
                print("current idx: ", idx + int((item-1)*per_class), "cate_id: ", item, end="\r")
                
            else:
                cur_name = categories[item-1]+str(idx)+".jpg"
                path = os.path.join(out_dir, cur_name)
                temp_im_info["file_name"] = cur_name
                temp_im_info["id"] = idx + 1 + int((item-1)*per_class)
                temp_im_info["height"] = im_or.shape[0]
                temp_im_info["width"] = im_or.shape[1]
                new_data_coco["images"].append(temp_im_info)
                cv2.imwrite(path, im_or)
                print("current idx: ", idx + int((item-1)*per_class), "cate_id: ", item, end="\r")
                
            
thread_pool = []
for item in range(1, class_num, 2):
    thread_pool.append(myThread(item, [item, item+1]))
thread1 = myThread(class_num, [class_num])
thread_pool.append(thread1) 


for item in thread_pool:
    item.start()

for item in thread_pool:
    item.join()
json.dump(new_data_coco,open(out_json_dir, "w"))
