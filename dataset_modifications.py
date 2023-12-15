"""The methods in this file convert the used datasets into one json file containing all annotation."""

import os
from pathlib import Path
import xmltodict
import json
from pprint import pprint

PASCALVOC_BASEPATH="/home/frank/datasets/VOC2012"
PASCALVOC_IMAGEPATH=PASCALVOC_BASEPATH+"/JPEGImages"
PASCALVOC_XML_ANNOTATIONPATH=PASCALVOC_BASEPATH+"/Annotations"
PASCALVOC_JSON_ANNOTATIONPATH=PASCALVOC_BASEPATH+"/JSONAnnotation"



def get_xml_annotation_path(image_path: Path):
    """gets the annotation for given image path"""
    return os.path.join(PASCALVOC_XML_ANNOTATIONPATH, path.name.replace(("jpg"),("xml")))

# conversion of pascal voc: many xml files to one json
one_json_annotation = {}
for path in Path(PASCALVOC_IMAGEPATH).rglob("*.jpg"):
    single_annotation = []
    with open(get_xml_annotation_path(path), 'r') as xmlfile:
        annotation = xmltodict.parse(xmlfile.read())
        if isinstance(annotation["annotation"]["object"], list):
            for object in annotation["annotation"]["object"]:
                strings = [object["bndbox"]["xmin"], object["bndbox"]["ymin"],object["bndbox"]["xmax"],object["bndbox"]["ymax"]]
                bbox = [int(string.split(".")[0]) for string in strings]  
                assert bbox[0]<bbox[2], f"Dimensional error for x {bbox[0]}, {bbox[2]}"
                assert bbox[1]<bbox[3], f"Dimensional error for y {bbox[1]}, {bbox[3]}"
                single_annotation.append({"label":object["name"], "bbox": bbox})
        else:
            bndbox = annotation["annotation"]["object"]["bndbox"]
            strings = [bndbox["xmin"], bndbox["ymin"],bndbox["xmax"],bndbox["ymax"]]
            bbox = [int(string.split(".")[0]) for string in strings]
            assert bbox[0]<bbox[2], "Dimensional ordering error for x"
            assert bbox[1]<bbox[3], "Dimensional ordering error for y"
            single_annotation.append({"label":annotation["annotation"]["object"]["name"], "bbox": bbox})
    one_json_annotation[path.as_posix()]=single_annotation

for k, (key, val) in enumerate(one_json_annotation.items()):#
    print(val)
    if k>10:
        break

len(one_json_annotation) # 17125
# one_json_annotation.keys() # keys are image paths
# one_json_annotation.values() # values are formatted: object: {xmin, ymin, xmax, ymax}
os.makedirs(PASCALVOC_JSON_ANNOTATIONPATH, exist_ok=True)
with open(os.path.join(PASCALVOC_JSON_ANNOTATIONPATH, "annotation.json"), 'w') as jsonoutfile:
    json.dump(one_json_annotation, jsonoutfile)

# set of all labels. tokenizer needs this file during __init__.py
all_labels = []
for anno in one_json_annotation.values():
    for single_anno in anno:
        # print(single_anno)
        all_labels.append(single_anno["label"])
labels = set(all_labels)
with open(os.path.join(PASCALVOC_JSON_ANNOTATIONPATH, "labels.json"), 'w') as jsonout:
    json.dump(list(labels), jsonout)


# MS COCO adjustment: bring it to the same format as above (two json files this time, one train, one val)
MS_COCO_IMG_PATH = "/home/frank/datasets/mscoco/images"
MS_COCO_ANNO_PATH = "/home/frank/datasets/mscoco/annotations"

# create label maps and labels file (requires labels.txt listing all 80 object classes)
with open("/home/frank/datasets/mscoco/annotations/labels.txt", 'r') as infile:
    labels = infile.readlines()

labelsdict = {0: "background"}
for k, label in enumerate(labels):
    labelsdict[k+1]=label.rstrip()
pprint(labelsdict)

with open("/home/frank/datasets/mscoco/annotations/labelmap.json", 'w') as mscocoout:
        json.dump(labelsdict, mscocoout)
labelslist = ["background"]

for label in labels:
    labelslist.append(label.rstrip())
with open("/home/frank/datasets/mscoco/annotations/labels.json", 'w') as mscocoout:
        json.dump(labelslist, mscocoout)


# build a proper label map to map int labels to string
with open("/home/frank/datasets/mscoco/annotations/labels.json", 'r') as mscocolabelin:
    coco_labels = json.load(mscocolabelin)
coco_labelmap = dict(zip(range(len(coco_labels)), coco_labels))

# now convert the annotation to the required format
for split in ["val2017", "train2017"]:
    with open(f"/home/frank/datasets/mscoco/annotations/instances_{split}.json", 'r') as mscocoin:
        coco = json.load(mscocoin)
    # first build a hash from img id to img path
    imgid_to_path = {}
    for img in coco["images"]:
        img_p = os.path.join("/home/frank/datasets/mscoco/images", split, img["file_name"])
        assert os.path.isfile(img_p), f"Not a valid img file: {img_p}"
        imgid_to_path[img["id"]]={"path": img_p, "anno":[]}
    for anno in coco["annotations"]:
        xmin, ymin, width, height = anno["bbox"]
        xmax = xmin + width
        ymax = ymin + height
        if (ymax>ymin+10) and (xmax>xmin+10):
            imgid_to_path[anno["image_id"]]["anno"].append({"label":coco_labelmap[anno["category_id"]], "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]})
    final_annotation = {val["path"]:val["anno"] for val in imgid_to_path.values()}
    with open(f"/home/frank/datasets/mscoco/annotations/budde_annotation_{split}.json", 'w') as mscocoout:
        json.dump(final_annotation, mscocoout)

# section for reduced class set (80 classes insteaf of 91)
# create label maps and labels file (requires labels.txt listing all 91 object classes)
with open("/home/frank/datasets/mscoco/annotations/labels_reduced.txt", 'r') as infile:
    labels = infile.readlines()
