import numpy as np
import torch
import json
from pprint import pprint
from PIL import Image
from typing import Any, Union, Callable
from pathlib import Path
from torchvision.transforms import Resize
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from transformers.image_processing_utils import BaseImageProcessor
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T


def read_json_annotation(filepath: Union[str, Path]) -> dict:
    "Reads a json file from path and returns its content."
    with open(filepath, "r") as jsonin:
        return json.load(jsonin)





class DatasetODT(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_path: Union[str, Path],
        preprocessor: BaseImageProcessor = None,
        training: bool = True,
        tokenizer: Callable = None,
    ) -> None:
        annotation = read_json_annotation(annotation_path)
        self.samples = list(annotation.keys())
        self.annotation = list(annotation.values())
        self.preprocessor = (
            preprocessor if preprocessor is not None else DeiTImageProcessor()
        )
        self.training = training
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> tuple:
        img = Image.open(self.samples[index])
        # TODO: the image preprocessor should probably go into the collate_fn() as it's set up to handle lists
        annotation = self.annotation[index]
        tokens = self.tokenizer(original_image_shape=img.size, annotation=annotation)
        return img, tokens

    def draw_patchwise_boundingboxes(self, img: Image, tokens: list) -> Image:
        labels = []
        boxes = []
        img = self.preprocessor(
            img, return_tensors="pt", do_rescale=False, do_normalize=False
        )
        img = img.data["pixel_values"][0, :, :, :].type(torch.uint8)

        for k in range(1, len(tokens) - 1, 3):
            labels.append(self.tokenizer.decode_labels(tokens[k]))
            ul_patch = tokens[k + 1]
            lr_patch = tokens[k + 2]
            xmin = ul_patch % (self.tokenizer.target_size / self.tokenizer.patch_size)
            xmax = lr_patch % (self.tokenizer.target_size / self.tokenizer.patch_size)
            ymin, _ = divmod(
                ul_patch, ((self.tokenizer.target_size / self.tokenizer.patch_size))
            )
            ymax, _ = divmod(
                lr_patch, ((self.tokenizer.target_size / self.tokenizer.patch_size))
            )
            bbox = torch.Tensor([xmin, ymin, xmax, ymax]) * self.tokenizer.patch_size
            boxes.append(bbox)
        drawn = draw_bounding_boxes(image=img, boxes=torch.stack(boxes), labels=labels)
        transform = T.ToPILImage()
        return transform(drawn)
