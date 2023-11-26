from pprint import pprint
import numpy as np
import random
from utils import read_json_annotation
import torch
from config import Config
import torch.nn.functional as F


class PatchwiseTokenizer:
    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        """Tokenizer mapping bounding box coordinates to image patch tokens.

        Args:
            verbose: Flag for debugging.
        """

        labels = read_json_annotation(Config.label_path)  # all labels in dataset
        self.target_size = Config.target_image_size  # after preprocessing (size, size)
        self.labelmap = dict(zip(labels, range(len(labels))))
        self.num_classes = len(labels)
        self.patch_size = Config.patch_size  # the size of patches
        self.num_patches = (self.target_size / self.patch_size) ** 2
        assert (
            self.target_size % self.patch_size == 0
        ), "Target image size does not match patch dimensions: target_size = n*patch_size"
        self.BOS = int(self.num_classes + self.num_patches)
        self.PAD = int(self.BOS + 1)
        self.EOS = int(self.PAD + 1)
        self.vocab_size = self.num_classes + self.num_patches + 3
        self.max_seq_len = Config.max_seq_len
        self.verbose = verbose
        if self.verbose:
            print(
                f"Initialized Tokenizer: BOS {self.BOS}, PAD {self.PAD}, EOS {self.EOS}."
            )
            print("Tokenizer classes:")
            pprint(self.labelmap)

    def __call__(self, original_image_shape: tuple, annotation: list) -> list:
        """Takes the bounding box annotation, returns sequence of tokens.
        Args:
            original_image_shape: tuple of image shape before resize.
            annotation: list of dicts containing label, bbox for all objects in img.

        Returns:
            List of tokens, starting with BOS and ending EOS."""
        width, height = original_image_shape
        tokens = [self.BOS]
        for anno in random.sample(annotation, len(annotation)):
            xmin, ymin, xmax, ymax = anno["bbox"]
            label = self.labelmap[anno["label"]]
            xmintoken, xmaxtoken = self.tokenize_bbox_dims(
                width, xmin, xmax
            )  # a bbox dimension turns into a patch number
            ymintoken, ymaxtoken = self.tokenize_bbox_dims(height, ymin, ymax)
            upper_left_token_id = self.get_token_id(xmintoken, ymintoken)
            lower_right_token_id = self.get_token_id(xmaxtoken, ymaxtoken)
            tokens.append(int(label))
            tokens.extend([int(upper_left_token_id), int(lower_right_token_id)])
        tokens.append(self.EOS)
        return torch.Tensor(tokens)

    def tokenize_bbox_dims(
        self, original_imagedim: int, minval: int, maxval: int
    ) -> tuple:
        """Turn original X/Y bbox dimension into corresponding token-ID.
        Args:
            original_imagedim: X/Y dimension of original image before resize.
            minval: bbox x/y min annotation.
            maxval: bbox x/y max annotation.
        Returns:
            tuple of upper left / lower right corner token.
        """
        mintoken, _ = divmod(
            (minval * self.target_size / original_imagedim), self.patch_size
        )
        maxtoken, remainder = divmod(
            (maxval * self.target_size / original_imagedim), self.patch_size
        )
        if remainder > 1:  # if bbox barely reaches into next patch
            maxtoken += 1

        return mintoken, maxtoken

    def get_token_id(self, x_token: int, y_token: int) -> int:
        """Determine token ID, taking into account bboxes that touch the image edges.
        Args:
            x_token: Tokenized bbox x-dimension.
            y_token: Tokenized bbox y-dimension.
        Returns:
            Token ID.
        Throws:
            AssertionError: If token-ID larger than self.num_patches."""
        if y_token == self.target_size / self.patch_size:
            y_token -= 1  # if the bbox extends into the edge
        token_id = x_token + y_token * (
            self.target_size / self.patch_size
        )  # each token has an ID in range(0, self.num_patches)
        assert (
            token_id <= self.num_patches
        ), f"Patch number violation token: {token_id}, from x, y: {x_token}, {y_token}"

        return token_id + len(
            self.labelmap
        )  # otherwise, tokens 0-len(labelmap) are double assigned!

    def decode_labels(self, val: int) -> str:
        """Returns the string label for a class token."""
        return [k for k, v in self.labelmap.items() if v == val]

    def decode_tokens(self, tokens: torch.Tensor, return_scores=False) -> dict:
        if return_scores:
            probs = torch.max(F.softmax(tokens, dim=0), dim=0).values
            tokens = torch.argmax(tokens, dim=1)
        batchsize, seq_len = tokens.shape
        tokens = (
            tokens.cpu().detach().numpy()
        )  # no cutting of EOS: is a prediction, BOS: already cut (299)
        sample_results = {bi: {"boxes": [], "labels": []} for bi in range(batchsize)}
        for k in range(0, seq_len - 3, 3):
            ul_patch = tokens[:, k + 1] - len(self.labelmap)
            lr_patch = tokens[:, k + 2] - len(self.labelmap)

            ymin_token, xmin_token = np.divmod(
                ul_patch, ((self.target_size / self.patch_size))
            )
            ymin = ymin_token * self.patch_size

            xmin = xmin_token * self.patch_size

            ymax_token, xmax_token = np.divmod(
                lr_patch, ((self.target_size / self.patch_size))
            )
            ymax_token = np.where(
                ymax_token < (self.target_size / self.patch_size),
                ymax_token + 1,
                ymax_token,
            )
            xmax_token = np.where(
                xmax_token == 0, (self.target_size / self.patch_size), xmax_token
            )

            ymax = ymax_token * self.patch_size
            xmax = xmax_token * self.patch_size

            for dim in [xmin, ymin, xmax, ymax]:
                assert np.less_equal(
                    dim, self.target_size
                ).any(), (
                    f"De-tokenized dimension {dim} exceeds imagesize {self.target_size}"
                )

            for batchindex in range(batchsize):
                if not tokens[batchindex, k] in [self.PAD, self.EOS]:
                    sample_results[batchindex]["boxes"].append(
                        torch.Tensor(
                            [
                                xmin[batchindex],
                                ymin[batchindex],
                                xmax[batchindex],
                                ymax[batchindex],
                            ]
                        ).unsqueeze(0)
                    )
                    sample_results[batchindex]["labels"].append(tokens[batchindex, k])

        for bi, dic in sample_results.items():
            if len(dic["boxes"]) == 0:
                sample_results[bi]["boxes"].append(
                    torch.Tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
                )
                sample_results[bi]["labels"].append(torch.Tensor([0.0]))
        if return_scores:
            return [
                {
                    "boxes": torch.cat(result["boxes"], dim=0),
                    "labels": torch.Tensor(result["labels"]).int(),
                    "scores": torch.ones_like(torch.Tensor(result["labels"])),
                }
                for result in sample_results.values()
            ]
        else:
            return [
                {
                    "boxes": torch.cat(result["boxes"], dim=0),
                    "labels": torch.Tensor(result["labels"]).int(),
                }
                for result in sample_results.values()
            ]
