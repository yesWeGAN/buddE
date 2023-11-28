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
    split='val',
    transforms=Config.train_transforms
)

ds.__getitem__(0)

dl = DataLoader(dataset=ds, batch_size=Config.batch_size, collate_fn=ds.collate_fn)

for k, (images, tokens) in enumerate(dl):
    print(f"Looking at batch {k}:")
    print(f"Images is a {type(images)} with shape: {images.shape}")
    print(f"Tokens is a {type(tokens)} with shape: {tokens.shape}")
    if k>2:
        break

""""
Images is a <class 'torch.Tensor'> with shape: torch.Size([16, 3, 224, 224])
Tokens is a <class 'torch.Tensor'> with shape: torch.Size([16, 23])
"""

# get probs from predictions 
import torch.nn.functional as F
import torch
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
    print(outputs.shape)    # 16, 198, 768  # with BOTTLENECK: 16,198,256
    # TODO: is there a class token for this? ie a distillation token? I do not think so, as 198 seems to be 14*14+EOS/BOS
    if k==1:
        break

decoder = Decoder(config=config)
summary(encoder.model, input_size=(3,224,224), device='cpu')
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
tokenvector = torch.randn((16,8))
tokenvector[:,:-1].shape

# try the avg meter
losses = [1.3456, 4.678, 2.9274]
avg = np.mean(np.array(losses))
avg

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

import torch
a = range(5)
b= range(5,10)
c = range(10, 15)
d = torch.Tensor([a,b,c])
# extended slicing: https://docs.python.org/release/2.3.5/whatsnew/section-slices.html
d[:, 0::2]  # selects every second row. if putting 0:3:2, select every second row until row 3



