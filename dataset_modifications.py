# script converts many XML files into one JSON file
import os
from pathlib import Path
import numpy as np
import xmltodict
import json

PASCALVOC_BASEPATH="/home/frank/datasets/VOC2012"
PASCALVOC_IMAGEPATH=PASCALVOC_BASEPATH+"/JPEGImages"
PASCALVOC_XML_ANNOTATIONPATH=PASCALVOC_BASEPATH+"/Annotations"
PASCALVOC_JSON_ANNOTATIONPATH=PASCALVOC_BASEPATH+"/JSONAnnotation"


def get_xml_annotation_path(image_path: Path):
    """gets the annotation for given image path"""
    return os.path.join(PASCALVOC_XML_ANNOTATIONPATH, path.name.replace(("jpg"),("xml")))


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
# one_json_annotation
# one_json_annotation.keys() # keys are image paths
# one_json_annotation.values() # values are formatted: object: {xmin, ymin, xmax, ymax}
os.makedirs(PASCALVOC_JSON_ANNOTATIONPATH, exist_ok=True)
with open(os.path.join(PASCALVOC_JSON_ANNOTATIONPATH, "annotation.json"), 'w') as jsonoutfile:
    json.dump(one_json_annotation, jsonoutfile)

# set of all labels
all_labels = []
for anno in one_json_annotation.values():
    for single_anno in anno:
        # print(single_anno)
        all_labels.append(single_anno["label"])
labels = set(all_labels)
with open(os.path.join(PASCALVOC_JSON_ANNOTATIONPATH, "labels.json"), 'w') as jsonout:
    json.dump(list(labels), jsonout)