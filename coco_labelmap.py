with open("/home/frank/datasets/mscoco/annotations/labels.txt", 'r') as infile:
    labels = infile.readlines()
import json
labelsdict = {0: "background"}
for k, label in enumerate(labels):
    labelsdict[k+1]=label.rstrip()

from pprint import pprint
pprint(labelsdict)
with open(f"/home/frank/datasets/mscoco/annotations/budde_annotation_labelmap.json", 'w') as mscocoout:
        json.dump(labelsdict, mscocoout)

labelslist = ["background"]
for label in labels:
    labelslist.append(label.rstrip())

with open(f"/home/frank/datasets/mscoco/annotations/budde_annotation_labels.json", 'w') as mscocoout:
        json.dump(labelslist, mscocoout)
