import json
import numpy as np
from tqdm import tqdm

test_num = 20000
json_dir = "/home/jovyan/fast-data/instances_default.json"
train_json_dir = "/home/jovyan/fast-data/train.json"
test_json_dir = "/home/jovyan/fast-data/test.json"

all_json = json.load(open(json_dir, "r"))

all_images = np.array(all_json["images"])
all_indexs = np.array(list(range(len(all_images))))
np.random.shuffle(all_indexs)
train_index = all_indexs[:-test_num]
test_index = all_indexs[-test_num:]

train_images = all_images[train_index]
test_images = all_images[test_index]

test_ids = []
for item in tqdm(test_images):
    test_ids.append(item["id"])

all_annotations = all_json["annotations"]
train_annotations = []
test_annotations = []
for item in tqdm(all_annotations):
    if item["image_id"] in test_ids:
        test_annotations.append(item)
    else:
        train_annotations.append(item)

train_json = {}
train_json["categories"] = all_json["categories"]
train_json["info"] = all_json["info"]
train_json["licenses"] = all_json["licenses"]
train_json["images"] = train_images.tolist()
train_json["annotations"] = train_annotations

test_json = {}
test_json["categories"] = all_json["categories"]
test_json["info"] = all_json["info"]
test_json["licenses"] = all_json["licenses"]
test_json["images"] = test_images.tolist()
test_json["annotations"] = test_annotations

json.dump(train_json, open(train_json_dir, "w"))
json.dump(test_json, open(test_json_dir, "w"))
