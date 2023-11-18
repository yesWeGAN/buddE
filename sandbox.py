# initialize the dataset

import json
import os
import dataset
import toml
import importlib
import tokenizer
import model
from torchsummary import summary

importlib.reload(dataset)
importlib.reload(tokenizer)
importlib.reload(model)
from torch.utils.data.dataloader import DataLoader
from dataset import DatasetODT
from tokenizer import PatchwiseTokenizer
from model import Encoder, Decoder
import numpy as np
import transformers.image_processing_utils

from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
a = 2
a**2

# load config
config = toml.load("config.toml")
type(config)    #dict

# setup the tokenizer to pass to the dataset
tokenizr = PatchwiseTokenizer(
    config=config
)

# setup the image processor
processor = DeiTImageProcessor()

# setup the dataset
ds = DatasetODT(
    config=config,
    preprocessor=processor,
    tokenizer=tokenizr,
)

dl = DataLoader(dataset=ds, batch_size=config["training"]["batch_size"], collate_fn=ds.collate_fn)

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
    AutoFeatureExtractor,
    DeiTForImageClassificationWithTeacher,
    DeiTFeatureExtractor,
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


# TOKEN DECODING SECTION FOR METRIC EVALUATION
def decode_tokens(tokens: torch.Tensor, return_scores = False, PAD=217, EOS=218) -> tuple:
    """This function needs to account for batched input, and that preds needs argmax first.
    This function does everything vectorized with numpy."""
    torch.save(tokens, "token_tensor_target.pt")      
    if return_scores:
        tokens = torch.argmax(tokens, dim=1)
    batchsize, seq_len = tokens.shape
    tokens = tokens.cpu().detach().numpy()  # no cutting of EOS: is a prediction, BOS: already cut (299)
    sample_results = {bi:{"boxes":[], "labels":[]} for bi in range(batchsize)}
    for k in range(0, seq_len-3, 3):
        ul_patch = tokens[:, k + 1] - len(tokenizr.labelmap)
        lr_patch = tokens[:, k + 2] - len(tokenizr.labelmap)

        ymin_token, xmin_token = np.divmod(
            ul_patch, ((tokenizr.target_size / tokenizr.patch_size))
        )
        ymin = ymin_token * tokenizr.patch_size

        xmin = xmin_token * tokenizr.patch_size

        ymax_token, xmax_token = np.divmod(
            lr_patch, ((tokenizr.target_size / tokenizr.patch_size))
        )
        ymax_token = np.where(ymax_token < (tokenizr.target_size / tokenizr.patch_size), ymax_token + 1, ymax_token)
        xmax_token = np.where(xmax_token == 0, (tokenizr.target_size / tokenizr.patch_size), xmax_token)


        ymax = ymax_token * tokenizr.patch_size
        xmax = xmax_token * tokenizr.patch_size

        for dim in [xmin, ymin, xmax, ymax]:
            assert (
                np.less_equal(dim, tokenizr.target_size).any()
            ), f"De-tokenized dimension {dim} exceeds imagesize {tokenizr.target_size}"
        for batchindex in range(batchsize):
            if not tokens[batchindex, k] in [tokenizr.PAD, tokenizr.EOS]:
                sample_results[batchindex]["boxes"].append(torch.Tensor([xmin[batchindex], ymin[batchindex], xmax[batchindex], ymax[batchindex]]).unsqueeze(0))
                sample_results[batchindex]["labels"].append(tokens[batchindex,k])
                # print(f"Appending batchindex {batchindex} and k {k}")

    if return_scores:
        return [{"boxes": torch.cat(result["boxes"], dim = 0), "labels": torch.Tensor(result["labels"]).int(), "scores":torch.ones_like(torch.Tensor(result["labels"]))} for result in sample_results.values()]
    else:
        return [{"boxes": torch.cat(result["boxes"], dim = 0), "labels": torch.Tensor(result["labels"]).int()} for result in sample_results.values()]

import torch
tensor = torch.load("token_tensor.pt").cpu().detach()
tokenizr.PAD
tokenizr.EOS
(tensor==int(tokenizr.EOS)).sum()
(tensor!=int(tokenizr.PAD)).sum()
tensor = torch.argmax(tensor, dim=0)
a = decode_tokens(tensor, return_scores=True)
len(a)
for b in a:
    if len(b["boxes"])>1:
        print(b)
tocat = [torch.Tensor([ 48.,  48., 128., 224.]), torch.Tensor([  0.,  32.,  32., 112.]), torch.Tensor([128., 144., 160., 224.])]
torch.cat(tocat, dim=0).shape


