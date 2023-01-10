import json
from tqdm import tqdm

dir_name = "1125_1.5w"

src_path = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/%s/annotations/ori_instances_default.json" % dir_name
dst_path = "/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled/%s/annotations/instances_default.json" % dir_name
# dst_path = "/home/jovyan/fast-data/temp.json"

src_file = open(src_path, 'r')
src_json = json.load(src_file)

dst_json = src_json
dst_images = []
dst_annotations = src_json["annotations"]
del_ids = []

for image in tqdm(src_json["images"]):
    image["file_name"] = "%s/data/" % dir_name + image["file_name"]
    if not os.path.exists(os.path.join("/home/jovyan/data-vol-polefs-1/dataset/sdj/labeled", image["file_name"])):
        del_ids.append(image["id"])
        print(image["file_name"])
    else:
        dst_images.append(image)

dst_json["images"] = dst_images
print(len(dst_images))

for annotation in tqdm(dst_annotations):
    if annotation["image_id"] in del_ids:
        dst_annotations.remove(annotation)

res = json.dumps(dst_json)
dst_file = open(dst_path, 'w')
dst_file.write(res)
