import torch
import json
from PIL import Image
from typing import Any, Union, Callable
from pathlib import Path
from torchvision.transforms import Resize
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from transformers.image_processing_utils import BaseImageProcessor


def read_json_annotation(filepath: Union[str, Path]):
    with open(filepath, "r") as jsonin:
        return json.load(jsonin)


class PatchwiseTokenizer:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Takes the bounding box annotation, returns all patches that contain the bounding box or parts of it.
        """
        #TODO: how to convert bounding box dimensions with the resizing operation?
        # it should be doable mathematically by just getting the scaling factor for each image in each dimension!
        # needs to be passed in. then: 
        # y_bbox = y_bbox * 224/y_max_original
        # y_patch = y_bbox % 16
        # patch_token = y_patch * 16 + x_patch
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
        self.tokenizer = tokenizer if tokenizer is not None else PatchwiseTokenizer()

    def __getitem__(self, index) -> tuple:
        img = Image.open(self.samples[index])
        # TODO: the image preprocessor should probably go into the collate_fn() as it's set up to handle lists
        annotation = self.annotation[index]
        tokens = self.tokenizer(annotation)
        return img, annotation
