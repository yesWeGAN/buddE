import torch
from PIL import Image
from typing import Union, Callable, Literal
from pathlib import Path
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from transformers.image_processing_utils import BaseImageProcessor
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
from torchvision.transforms import Compose
from utils import read_json_annotation
from torch.nn.utils.rnn import pad_sequence


class DatasetODT(torch.utils.data.Dataset):
    def __init__(
        self,
        config: dict,
        preprocessor: BaseImageProcessor = None,
        tokenizer: Callable = None,
        transforms: Union[Compose, None] = None,
        split: Literal["train","val", None] = None,
        split_ratio: float = 0.8,
    ) -> None:
        """Dataset class for an object detection Transformer.
        Args:
            config: The config parsed from toml file.
            preprocessor: Image preprocessor to scale/normalize images.
            tokenizer: Tokenizer to tokenize annotation.
            transforms: Compose of transforms to apply to the images. Must return torch.Tensor."""
        self.annotation_path = config["data"]["annotation_path"]
        
        if split:
            annotation = read_json_annotation(self.annotation_path)
            num_samples = len(annotation)
            self.samples = list(annotation.keys())[:int(num_samples*split_ratio)] if split=="train" else list(annotation.keys())[int(num_samples*split_ratio):]
            self.annotation = list(annotation.values())[:int(num_samples*split_ratio)] if split=="train" else list(annotation.values())[int(num_samples*split_ratio):]

        else:
            self.samples = None
            self.annotation = None

        self.preprocessor = (
            preprocessor if preprocessor is not None else DeiTImageProcessor()
        )
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __getitem__(self, index) -> tuple:
        assert self.samples is not None, "No samples in dataset. Make sure to define dataset split [train | val]."
        img = Image.open(self.samples[index])
        if self.transforms:
            img = self.transforms(img) 
        # TODO: if not self.training: return only img, no tokens
        annotation = self.annotation[index]
        tokens = self.tokenizer(original_image_shape=img.size, annotation=annotation)

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
        tokens = pad_sequence(sequences=tokens, padding_value=self.tokenizer.PAD, batch_first=True)
        
        # disable the following lines to not pad batch to MAX_SEQ_LEN
        pad_tokens = torch.ones((tokens.shape[0], self.tokenizer.max_seq_len-tokens.shape[1])).fill_(self.tokenizer.PAD)
        tokens = torch.cat((tokens, pad_tokens), dim=1).long()
        assert tokens.shape[0]==16, "Batch dimension is off after padding tokens."
        assert tokens.shape[1]==300, "MAX_SEQ_LEN padding of tokens failed."
        
        images = self.preprocessor(images, return_tensors = 'pt')
        return images.data['pixel_values'], tokens

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

