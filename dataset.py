from typing import Callable, Union

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose
from torchvision.utils import draw_bounding_boxes
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor

from config import Config
from utils import read_json_annotation


class DatasetODT(torch.utils.data.Dataset):
    def __init__(
        self,
        preprocessor: BaseImageProcessor = None,
        tokenizer: Callable = None,
    ) -> None:
        """Dataset class for an object detection Transformer.
        Args:
            preprocessor: Image preprocessor to scale/normalize images.
            tokenizer: Tokenizer to tokenize annotation.
            transforms: Compose of transforms to apply to the images. Must return torch.Tensor.
        """
        self.annotation_path = None
        self.preprocessor = (
            preprocessor if preprocessor is not None else DeiTImageProcessor()
        )
        self.tokenizer = tokenizer
        self.transforms = None

    def get_train(self, transforms: Union[Compose, A.Compose] = None):
        self.annotation_path = Config.train_annotation_path
        self.transforms = transforms
        anno = read_json_annotation(self.annotation_path)
        num_samples = len(anno)
        if Config.split_ratio:
            self.samples = list(anno.keys())[: int(num_samples * Config.split_ratio)]
            self.annotation = list(anno.values())[
                : int(num_samples * Config.split_ratio)
            ]
        else:
            self.samples = list(anno.keys())
            self.annotation = list(anno.values())
            print(f"COCO: Not splitting  {num_samples} train samples.")
        return self

    def get_val(self, transforms: Union[Compose, A.Compose] = None):
        self.annotation_path = Config.val_annotation_path
        self.transforms = transforms
        anno = read_json_annotation(self.annotation_path)
        num_samples = len(anno)
        if Config.split_ratio:
            self.samples = list(anno.keys())[int(num_samples * Config.split_ratio) :]
            self.annotation = list(anno.values())[
                int(num_samples * Config.split_ratio) :
            ]
        else:
            print(f"COCO: Not splitting  {num_samples} val samples.")
            self.samples = list(anno.keys())
            self.annotation = list(anno.values())
        return self

    def __getitem__(self, index) -> tuple:
        img = Image.open(self.samples[index])
        if img.mode != "RGB":
            img = img.convert("RGB")
        annotation = self.annotation[index]
        bboxes = [anno["bbox"] for anno in annotation]
        labels = [anno["label"] for anno in annotation]

        if self.transforms:
            img = np.array(img)
            transformed = self.transforms(image=img, bboxes=bboxes, class_labels=labels)
            img = Image.fromarray(transformed["image"])
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        tokens = self.tokenizer(
            original_image_shape=img.size, bboxes=bboxes, labels=labels
        )

        return img, tokens

    def __len__(self) -> int:
        return len(self.samples)

    def collate_fn(self, batch: tuple) -> tuple:
        """Collate the batches of images and token sequences into padded tensors.
        Args:
            batch: tuple of images, tokens. tokens needs to be torch.Tensor
            max_len: Maximum sequence length to pad to.

        Returns:
            images: torch.Tensor with shape [BATCH, CHANNELS, IMAGEDIM, IMAGEDIM].
            tokens: torch.Tensor with shape [BATCH, MAX_PADDED_SEQ_LEN].
        """

        images = []
        tokens = []
        for img, seq in batch:
            images.append(img)
            tokens.append(seq)
        tokens = pad_sequence(
            sequences=tokens, padding_value=self.tokenizer.PAD, batch_first=True
        )

        # disable the following lines to not pad batch to MAX_SEQ_LEN
        pad_tokens = torch.ones(
            (tokens.shape[0], self.tokenizer.max_seq_len - tokens.shape[1])
        ).fill_(self.tokenizer.PAD)
        tokens = torch.cat((tokens, pad_tokens), dim=1).long()
        # assert tokens.shape[0]==16, "Batch dimension is off after padding tokens."
        assert tokens.shape[1] == 300, "MAX_SEQ_LEN padding of tokens failed."

        images = self.preprocessor(images, return_tensors="pt")
        return images.data["pixel_values"], tokens

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
