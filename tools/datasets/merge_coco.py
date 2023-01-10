import json


# j1_dir = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/1109_1w/annotations/instances_default.json"
# j2_dir = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/1125_1.5w/annotations/instances_default.json"

j1_dir = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/1107_9w/annotations/instances_default.json"
j2_dir = "/home/jovyan/fast-data/temp.json"

j1 = json.load(open(j1_dir, "r"))
j2 = json.load(open(j2_dir, "r"))
    
new_dir = "/home/jovyan/fast-data/instances_default.json"

print(j1["categories"]==j2["categories"])

if not j1["categories"] == j2["categories"]:
    print("error categories")
    exit()
    cat_dic1 = {item["name"]:item["id"] for item in j1["categories"]}
    cat_dic2 = {item["id"]:item["name"] for item in j2["categories"]}
    for item in j2["annotations"]:
        item["category_id"] = cat_dic1[cat_dic2[item["category_id"]]]

print("j1 images len", len(j1["images"]))
print("j1 annotations len", len(j1["annotations"]))
print("j2 images len", len(j2["images"]))
print("j2 annotations len", len(j2["annotations"]))
        
j1_images_len = j1["images"][-1]["id"] + 10
j1_annotations_len = j1["annotations"][-1]["id"] + 10
for item in j2["images"]:
    item["id"] += j1_images_len
for item in j2["annotations"]:
    item["id"] += j1_annotations_len
for item in j2["annotations"]:
    item["image_id"] += j1_images_len
    
j1["images"] += j2["images"]
j1["annotations"] += j2["annotations"]

print("new images len", len(j1["images"]))
print("new annotations len", len(j1["annotations"]))
json.dump(j1, open(new_dir, "w"))