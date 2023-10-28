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
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import PILToTensor, Compose

def collate_fn(batch: tuple, max_len: int = None, pad_token: int = None, img_preprocessor: Callable = None) -> tuple:
    """Collate the batches of images and token sequences into padded tensors.
    Args:
        batch: tuple of images, tokens. tokens needs to be torch.Tensor
        max_len: Maximum sequence length to pad to.
        pad_token: The token to pad the sequences with. Usually defined by tokenizer."""
    images = []
    tokens = []
    for img, seq in batch:
        images.append(img)
        tokens.append(seq)
    tokens = pad_sequence(sequences=tokens)



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
        collate_fn: Callable = collate_fn,
        transforms: Union[Compose, None] = None
    ) -> None:
        """Dataset class for an object detection Transformer.
        Args:
            annotation_path: The path to a json-file containing annotation for dataset.
            preprocessor: Image preprocessor to scale/normalize images.
            training: flag for training to control if labels are returned. Defaults to True.
            tokenizer: Tokenizer to tokenize annotation.
            collate_fn: Collate function for the dataloader.
            transforms: Compose of transforms to apply to the images. Must return torch.Tensor."""
        annotation = read_json_annotation(annotation_path)
        self.samples = list(annotation.keys())
        self.annotation = list(annotation.values())
        self.preprocessor = (
            preprocessor if preprocessor is not None else DeiTImageProcessor()
        )
        self.training = training
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.transforms = transforms

    def __getitem__(self, index) -> tuple:
        img = Image.open(self.samples[index])
        if self.transforms:
            img = self.transforms(img) 
        # TODO: the image preprocessor should probably go into the collate_fn() as it's set up to handle lists
        annotation = self.annotation[index]
        tokens = self.tokenizer(original_image_shape=img.size, annotation=annotation)

        return img, tokens

    def __len__(self) -> int:
        return len(self.samples)

    def draw_patchwise_boundingboxes(self, img: Image, tokens: list) -> Image:
        """Draws the patch-wise bounding box on the image.

        Args:
            img: PIL.Image that will be drawn on
            tokens: the sequence of tokens from the tokenizer.

        Returns:
            PIL.Image"""
        labels = []
        boxes = []
        img = self.preprocessor(
            img, return_tensors="pt", do_rescale=False, do_normalize=False
        )
        img = img.data["pixel_values"][0, :, :, :].type(torch.uint8)
        labels, boxes = self.tokenizer.decode_tokens(tokens)
        drawn = draw_bounding_boxes(image=img, boxes=torch.stack(boxes), labels=labels)
        transform = T.ToPILImage()
        return transform(drawn)
