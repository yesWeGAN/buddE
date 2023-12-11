# initialize the dataset

import dataset
import toml
import importlib
import tokenizer
import model
from torchsummary import summary
import config

importlib.reload(dataset)
importlib.reload(tokenizer)
importlib.reload(model)
importlib.reload(config)
from torch.utils.data.dataloader import DataLoader
from dataset import DatasetODT
from tokenizer import PatchwiseTokenizer
from model import Encoder, Decoder
import numpy as np
from config import Config

from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor

a = 2
a**2

# setup the tokenizer to pass to the dataset
tokenizr = PatchwiseTokenizer()

# setup the image processor
processor = DeiTImageProcessor()

# setup the dataset
ds = DatasetODT(
    preprocessor=processor,
    tokenizer=tokenizr,
    split="val",
    transforms=Config.train_transforms,
)

ds.__getitem__(0)

dl = DataLoader(dataset=ds, batch_size=Config.batch_size, collate_fn=ds.collate_fn)

for k, (images, tokens) in enumerate(dl):
    print(f"Looking at batch {k}:")
    print(f"Images is a {type(images)} with shape: {images.shape}")
    print(f"Tokens is a {type(tokens)} with shape: {tokens.shape}")
    if k > 2:
        break

""""
Images is a <class 'torch.Tensor'> with shape: torch.Size([16, 3, 224, 224])
Tokens is a <class 'torch.Tensor'> with shape: torch.Size([16, 23])
"""

from torchvision.datasets import ImageFolder

folderset = ImageFolder("inference_imgs")
vars(folderset)
# get probs from predictions
import torch.nn.functional as F
import torch

rando = torch.rand((32, 600, 300))
F.softmax(rando, dim=0).shape
torch.max(F.softmax(rando, dim=0), dim=0).values.shape
probs = None

a = torch.load("224_predicted.pt")
b = torch.load("224_truth.pt")
a = tokenizr.decode_tokens(a, return_scores=True)
c = tokenizr.decode_tokens(b)

for result in a:
    print(len(result["labels"]))


for result in c:
    print(result)

a.shape
preds = torch.argmax(F.softmax(a, dim=1), dim=1)
preds.shape
preds
b
import math

(1 / math.sqrt(256))

# next up: we need the model
encoder = Encoder(config=config)
for k, (images, tokens) in enumerate(dl):
    outputs = encoder(images)
    print(outputs.shape)  # 16, 198, 768  # with BOTTLENECK: 16,198,256
    # TODO: is there a class token for this? ie a distillation token? I do not think so, as 198 seems to be 14*14+EOS/BOS
    if k == 1:
        break

decoder = Decoder(config=config)
summary(encoder.model, input_size=(3, 224, 224), device="cpu")
encoder.model.device


# load some samples
for k in range(len(ds)):
    img, anno = ds.__getitem__(k)
    annotated = ds.draw_patchwise_boundingboxes(img, anno)
    annotated.save(f"bboxdrawn_{k}.jpg")

tokenizer(original_image_shape=img.size, annotation=anno)

# using the DeiTImageProcessor on some images
from transformers import (
    DeiTModel,
)

# try around with the tokens for input/output
tokenvector = torch.randn((16, 8))
tokenvector[:, :-1].shape

# try the avg meter
losses = [1.3456, 4.678, 2.9274]
avg = np.mean(np.array(losses))
avg

# see the resizing operation of DeiTImageProcessor
processor = DeiTImageProcessor()
preprocessed_image = processor(img, return_tensors="pt")
type(preprocessed_image)
preprocessed_image.data["pixel_values"][0, :, :, :].astype()
import torch

torch.IntTensor
model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
outputs = model(**preprocessed_image)
outputs.last_hidden_state.shape  # [1, 198, 768]. image size 224,224 gives 14 patches, 14*14 = 198 + BOS + EOS


processor = DeiTImageProcessor(
    do_rescale=False, do_normalize=False
)  # does (3 ,224, 224)
a = processor.preprocess(img)["pixel_values"][0]
a.shape
from PIL import Image
import numpy as np

image = Image.fromarray(np.moveaxis(a, 0, 2), mode="RGB")
image.save("after_resize.jpg")
img.save("before_resize.jpg")

import torch

a = range(5)
b = range(5, 10)
c = range(10, 15)
d = torch.Tensor([a, b, c])
# extended slicing: https://docs.python.org/release/2.3.5/whatsnew/section-slices.html
d[
    :, 0::2
]  # selects every second row. if putting 0:3:2, select every second row until row 3

mARs = [
    0.1223,
    0.4430,
    0.3541,
    0.5258,
    0.0991,
    0.5692,
    0.2936,
    0.0581,
    0.1650,
    0.1402,
    0.2551,
    0.3260,
    0.2762,
    0.2339,
    0.3613,
    0.2959,
    0.3047,
    0.1571,
    0.0869,
    0.1798,
]
for k, mar in enumerate(mARs):
    print(f"{tokenizr.decode_labels(k)[0]}: {mar}")

# try out the dataset class with MS COCO
from config import Config
from dataset import DatasetODT

from tokenizer import PatchwiseTokenizer
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor

# setup tokenizer
tokenizr = PatchwiseTokenizer()

# setup the image processor
processor = DeiTImageProcessor(
    size={
        "height": Config.target_image_size,
        "width": Config.target_image_size,
    },
    do_center_crop=False,
)

# setup the dataset
ds = DatasetODT(
    preprocessor=processor,
    tokenizer=tokenizr,
    transforms=Config.train_transforms,
    split="train",
)
for k in range(len(ds)):
    img, anno = ds.__getitem__(k)
    assert img.mode == "RGB", f"This one violates RGB: {k}"
    if k > 10:
        break

import torch

coco_results = {
    "classes": torch.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        ],
    ),
    "map": torch.Tensor([0.1854]),
    "map_50": torch.Tensor([0.2879]),
    "map_75": torch.Tensor([0.1914]),
    "map_large": torch.Tensor([0.3375]),
    "map_medium": torch.Tensor([0.0964]),
    "map_per_class": torch.Tensor(
        [
            0.0000,
            0.1587,
            0.0807,
            0.0850,
            0.1466,
            0.4280,
            0.3142,
            0.5210,
            0.1001,
            0.0663,
            0.1023,
            0.4772,
            0.4456,
            0.1694,
            0.1448,
            0.1847,
            0.5027,
            0.4522,
            0.2335,
            0.1282,
            0.1364,
            0.3399,
            0.5849,
            0.3493,
            0.4781,
            0.0260,
            0.1057,
            0.0178,
            0.2249,
            0.0727,
            0.4131,
            0.2191,
            0.3085,
            0.2213,
            0.1496,
            0.1279,
            0.1619,
            0.3278,
            0.2381,
            0.2812,
            0.0191,
            0.0376,
            0.0381,
            0.0868,
            0.0324,
            0.0241,
            0.0599,
            0.0253,
            0.0240,
            0.2047,
            0.0942,
            0.0409,
            0.0248,
            0.0425,
            0.2116,
            0.0688,
            0.0620,
            0.0392,
            0.2283,
            0.0351,
            0.3820,
            0.1023,
            0.4415,
            0.2425,
            0.2200,
            0.3192,
            0.1036,
            0.2291,
            0.1536,
            0.2804,
            0.1985,
            0.0000,
            0.2258,
            0.2338,
            0.0142,
            0.2990,
            0.1234,
            0.1939,
            0.1946,
            0.0000,
            0.1375,
        ]
    ),
    "map_small": torch.Tensor([0.0261]),
    "mar_1": torch.Tensor([0.2150]),
    "mar_10": torch.Tensor([0.2775]),
    "mar_100": torch.Tensor([0.2797]),
    "mar_100_per_class": torch.Tensor(
        [
            0.0000,
            0.3516,
            0.1895,
            0.2471,
            0.2858,
            0.5430,
            0.4229,
            0.6037,
            0.1514,
            0.2014,
            0.1895,
            0.6033,
            0.5119,
            0.3382,
            0.2038,
            0.2863,
            0.6089,
            0.6009,
            0.4238,
            0.2986,
            0.2595,
            0.4914,
            0.6479,
            0.4922,
            0.5996,
            0.0451,
            0.1974,
            0.0271,
            0.3237,
            0.1956,
            0.4863,
            0.3085,
            0.3811,
            0.3377,
            0.2716,
            0.2038,
            0.2180,
            0.4434,
            0.3771,
            0.3567,
            0.1194,
            0.1291,
            0.1156,
            0.1263,
            0.0591,
            0.0409,
            0.1470,
            0.1497,
            0.0716,
            0.2983,
            0.2280,
            0.1510,
            0.1430,
            0.1347,
            0.3435,
            0.1932,
            0.2151,
            0.1507,
            0.3287,
            0.0886,
            0.5184,
            0.1605,
            0.6190,
            0.3042,
            0.3069,
            0.4021,
            0.1661,
            0.3260,
            0.2405,
            0.3073,
            0.2592,
            0.0000,
            0.3219,
            0.2817,
            0.1129,
            0.3971,
            0.2085,
            0.2167,
            0.3607,
            0.0000,
            0.1898,
        ]
    ),
    "mar_large": torch.Tensor([0.4423]),
    "mar_medium": torch.Tensor([0.1686]),
    "mar_small": torch.Tensor([0.0374]),
}

maps = []
map_50s = []
# filter the predictions and calculate mAP again
# first, get the map from index to label
with open("/home/frank/datasets/mscoco/annotations/labels.txt", 'r') as infile:
    labels = infile.readlines()
labelsdict = {0: "background"}
for k, label in enumerate(labels):
    labelsdict[k+1]=label.rstrip()

# now, get the labels from val2017
with open("/home/frank/datasets/mscoco/annotations/labels_reduced.txt", 'r') as infile:
    labels = infile.readlines()
val_labels = [label.rstrip() for label in labels]

# finally, through all predicted classes, check if that index maps to a class in val2017 challenge
for k, index in enumerate(coco_results["classes"].numpy()):
    if coco_results["map_per_class"][k]<0.05:
        print(f"This label has a tiny map: {labelsdict[index]}")
        maps.append(coco_results["map_per_class"][k])

len(maps)   # it's 80. probably, only those 80 exist in validation set?

coco_results["map_per_class"].shape