import json

src_path = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/1125_1.5w/annotations/instances_default.json"
dst_path = "/home/jovyan/fast-data/temp.json"

src_file = open(src_path, 'r')
src_json = json.load(src_file)

target_labels = {"底座": 1, "体重秤": 2, "鞋子": 3, "插线板": 4, "线团线材": 5, "宠物粪便": 6, "袜子毛巾": 7, "玩具": 8, "垃圾桶": 9, "塑料袋": 10,
                 "花盆": 11}

error_map = {}
for category, target_label in zip(src_json["categories"], target_labels.keys()):
    error_map[category["id"]] = category["name"]
    print(category)
    category["name"] = target_label
    print(category)

print(error_map)

for annotation in src_json["annotations"]:
    annotation["category_id"] = target_labels[error_map[annotation["category_id"]]]
#     print(annotation["category_id"], "->", target_labels[error_map[annotation["category_id"]]])

res = json.dumps(src_json)
dst_file = open(dst_path, 'w')
dst_file.write(res)
