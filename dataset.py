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


def read_json_annotation(filepath: Union[str, Path]) -> dict:
    "Reads a json file from path and returns its content."
    with open(filepath, "r") as jsonin:
        return json.load(jsonin)


class PatchwiseTokenizer:
    def __init__(
        self,
        label_path: Union[Path, str],
        target_size: int,
        patch_size: int,
        verbose: bool = False,
    ) -> None:
        """Tokenizer mapping bounding box coordinates to image patch tokens.
        Args:
            labels: json-filepath containing all labels in dataset.
            target_size: size of image after preprocessing (size, size).
            patchsize: The size of patches (patchsize, patchsize) each image is decomposed to (e.g. 224x224 -> 14x14 patches of (16,16))
            verbose: Flag for debugging.
        """
        assert target_size%patch_size==0, "Target image size does not match patch dimensions: target_size = n*patch_soue"
        labels = read_json_annotation(label_path)
        self.target_size = target_size
        self.labelmap = dict(zip(labels, range(len(labels))))
        self.num_classes = len(labels)
        self.patch_size = patch_size
        self.num_patches = (target_size/patch_size)**2
        self.BOS = self.num_classes + self.num_patches
        self.PAD = self.BOS + 1
        self.EOS = self.PAD + 1
        self.vocab_size = self.num_classes + self.num_patches + 3
        self.verbose = verbose
        if self.verbose:
            print(
                f"Initialized Tokenizer: BOS {self.BOS}, PAD {self.PAD}, EOS {self.EOS}."
            )
            print("Tokenizer classes:")
            pprint(self.labelmap)

    def __call__(self, original_image_shape: tuple, annotation: dict) -> Any:
        """Takes the bounding box annotation, returns all patches that contain the bounding box or parts of it."""
        # TODO: how to convert bounding box dimensions with the resizing operation?
        # it should be doable mathematically by just getting the scaling factor for each image in each dimension!
        # needs to be passed in. then:
        # y_bbox = y_bbox * 224/y_max_original
        # y_patch = y_bbox % 16
        # patch_token = y_patch * 16 + x_patch
        width, height = original_image_shape
        arrays = []
        for anno in annotation:
            bbox = np.array(anno["bbox"], dtype=int)
            label = self.labelmap(anno["label"])
            arr = np.array()
            print(bbox)
            print(label)
        # TODO: nope! I need to seperate the xmins from the ymins and calculate the tokens, not return tensors!

        # arrays = [np.array(sample["bbox"].values()) for sample in anno]
        # TODO: store the bboxes as a list of integers, saves a lot of time!
        pass


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
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else PatchwiseTokenizer(
                labels=set(["ass", "tits"]),
                target_size=(224, 224),
                num_patches=196,
                verbose=True,
            )
        )

    def __getitem__(self, index) -> tuple:
        img = Image.open(self.samples[index])
        # TODO: the image preprocessor should probably go into the collate_fn() as it's set up to handle lists
        annotation = self.annotation[index]
        tokens = self.tokenizer(original_image_shape=img.size, annotation=annotation)
        return img, annotation
