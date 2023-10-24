# initialize the dataset

import json
import os
import dataset
import toml
import importlib
import tokenizer

importlib.reload(dataset)
importlib.reload(tokenizer)
from dataset import DatasetODT
from tokenizer import PatchwiseTokenizer
import numpy as np

from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor


# load config
config = toml.load("config.toml")

# setup the tokenizer to pass to the dataset
tokenizer = PatchwiseTokenizer(
    label_path=config["data"]["label_path"],
    target_size=config["transforms"]["target_image_size"],
    patch_size=config["transforms"]["patch_size"],
)

# setup the image processor
processor = DeiTImageProcessor()

# setup the dataset
ds = DatasetODT(
    annotation_path=config["data"]["annotation_path"],
    preprocessor=processor,
    training=True,
    tokenizer=tokenizer,
)

# load some samples
for k in range(len(ds)):
    img, anno = ds.__getitem__(k)
    annotated = ds.draw_patchwise_boundingboxes(img, anno)
    annotated.save(f"bboxdrawn_{k}.jpg")

tokenizer(original_image_shape=img.size, annotation=anno)

# using the DeiTImageProcessor on some images
from transformers import (
    AutoFeatureExtractor,
    DeiTForImageClassificationWithTeacher,
    DeiTFeatureExtractor,
    DeiTModel,
)

# see the resizing operation of DeiTImageProcessor
processor = DeiTImageProcessor()
preprocessed_image = processor(img, return_tensors="pt")
type(preprocessed_image)
preprocessed_image.data["pixel_values"][0, :,:,:].astype()
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

"""
Summary: 
model has DeiTEncoder + embedding layer
embeddinger layer: DeiTEmbeddings, which uses DeiTPatchEmbeddings:
    a Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16)) projecting as follows:
This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
`hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
Transformer.
which essentially means: 
the tokenization of the image is a learnt convolutional operation.
"""
