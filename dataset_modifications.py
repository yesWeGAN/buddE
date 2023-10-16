# script converts many XML files into one JSON file
import os
from pathlib import Path
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
                single_annotation.append({object["name"]:object["bndbox"]})
        else:
            single_annotation.append({annotation["annotation"]["object"]["name"]:annotation["annotation"]["object"]["bndbox"]})
    one_json_annotation[path.as_posix()]=single_annotation

len(one_json_annotation) # 17125

one_json_annotation.keys() # keys are image paths
one_json_annotation.values() # values are formatted: object: {xmin, ymin, xmax, ymax}
os.makedirs(PASCALVOC_JSON_ANNOTATIONPATH, exist_ok=True)
with open(os.path.join(PASCALVOC_JSON_ANNOTATIONPATH, "annotation.json"), 'w') as jsonoutfile:
    json.dump(one_json_annotation, jsonoutfile)