from pprint import pprint
from typing import Union
from pathlib import Path
import json

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
        assert (
            target_size % patch_size == 0
        ), "Target image size does not match patch dimensions: target_size = n*patch_soue"
        labels = read_json_annotation(label_path)
        self.target_size = target_size
        self.labelmap = dict(zip(labels, range(len(labels))))
        self.num_classes = len(labels)
        self.patch_size = patch_size
        self.num_patches = (target_size / patch_size) ** 2
        self.BOS = int(self.num_classes + self.num_patches)
        self.PAD = int(self.BOS + 1)
        self.EOS = int(self.PAD + 1)
        self.vocab_size = self.num_classes + self.num_patches + 3
        self.verbose = verbose
        if self.verbose:
            print(
                f"Initialized Tokenizer: BOS {self.BOS}, PAD {self.PAD}, EOS {self.EOS}."
            )
            print("Tokenizer classes:")
            pprint(self.labelmap)

    def __call__(self, original_image_shape: tuple, annotation: dict) -> list:
        """Takes the bounding box annotation, returns sequence of tokens."""
        width, height = original_image_shape
        tokens = [self.BOS]
        for anno in annotation:
            xmin, ymin, xmax, ymax = anno["bbox"]
            label = self.labelmap[anno["label"]]
            xmintoken, xmaxtoken = self.tokenize_bbox_dims(
                width, xmin, xmax
            )  # a bbox dimension turns into a patch number
            ymintoken, ymaxtoken = self.tokenize_bbox_dims(height, ymin, ymax)
            upper_left_token_id = xmintoken + ymintoken * (
                self.target_size / self.patch_size
            )  # each token has an ID in range(0, self.num_patches)
            lower_right_token_id = xmaxtoken + ymaxtoken * (
                self.target_size / self.patch_size
            )
            assert (
                upper_left_token_id <= self.num_patches
            ), f"Patch number violation upper left: {upper_left_token_id}, from xmin, ymin {xmin}, {ymin}, mintoken, maxtoken {xmintoken}, {ymintoken}"
            assert (
                lower_right_token_id <= self.num_patches
            ), f"Patch number violation lower right: {lower_right_token_id}, from xmax, ymax {xmax}, {ymax}, mintoken, maxtoken {xmaxtoken}, {ymaxtoken}"
            tokens.append(int(label))
            tokens.extend([int(upper_left_token_id), int(lower_right_token_id)])
        tokens.append(self.EOS)
        return tokens

    def tokenize_bbox_dims(self, original_imagedim: int, minval: int, maxval: int):
        """List comprehension to get modulo division (patch)"""
        mintoken, _ = divmod(
            (minval * self.target_size / original_imagedim), self.patch_size
        )
        maxtoken, remainder = divmod(
            (maxval * self.target_size / original_imagedim), self.patch_size
        )
        if remainder > 1:  # if bbox barely reaches into next patch
            maxtoken += 1

        if maxtoken >= (self.target_size / self.patch_size):
            maxtoken -= 1
        # TODO: this leads to the borders never being part of the bbox. I need to adjust the calculation of the token ID, not the token itself
        return mintoken, maxtoken

    def decode_labels(self, val: int) -> str:
        """Returns the string label for a class token."""
        return [k for k, v in self.labelmap.items() if v == val]