from pprint import pprint
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
from transformers.image_processing_utils import BaseImageProcessor
from config import Config
from tokenizer import PatchwiseTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path


class DatasetInference:
    """Dataset for inference only. Does not handle any annotation."""
    def __init__(
        self,
        root: str,
        preprocessor: BaseImageProcessor,
        tokenizer: PatchwiseTokenizer,
        valid_types=["*.png", "*.jpg", "*.jpeg"],
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.valid_types = valid_types
        self.root = Path(root)
        self.samples = self._find_images()

    def _find_images(self):
        samples = []
        for filetype in self.valid_types:
            samples.extend(list(Path(self.root).rglob(filetype)))
        return samples

    def __getitem__(self, index) -> tuple:
        """Returns the PIL.Image for processing and the filepath for plotting."""
        assert self.samples is not None, "No samples in dataset."
        img = Image.open(self.samples[index])
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, self.samples[index]

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch: tuple) -> tuple:
        """Regular collate_fn without tokenization."""

        images = []
        paths = []
        for img, path in batch:
            images.append(img)
            paths.append(path)
        images = self.preprocessor(images, return_tensors="pt")
        return images.data["pixel_values"], paths

    def draw_patchwise_boundingboxes(
        self,
        img: Image,
        predictions: dict,
    ) -> Image:
        """Draws the patch-wise bounding box on the image.

        Args:
            img: PIL.Image that will be drawn on
            predictions: Dict of results.

        Returns:
            PIL.Image"""
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.preprocessor(
            img, return_tensors="pt", do_rescale=False, do_normalize=False
        )
        img = img.data["pixel_values"][0, :, :, :].type(torch.uint8)
        drawn = draw_bounding_boxes(
            image=img,
            boxes=predictions["boxes"],
            labels=[
                self.tokenizer.decode_labels(label) for label in predictions["labels"]
            ],
            font='Ubuntu-M',
            font_size=25
        )
        transform = T.ToPILImage()
        return transform(drawn)


class ModelInference:
    """This class handles the inference with a trained model."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: DatasetInference,
    ):
        self.batch_size = Config.validation_batch_size
        self.num_workers = Config.num_workers
        self.model = model.eval()
        self.val_dl = self._setup_dl(ds=dataset)
        self.tokenizer = dataset.tokenizer
        self.predictions = {}
        self.dataset = dataset

    def _setup_dl(self, ds: DatasetInference) -> tuple:
        """Takes a dataset and prepares DataLoaders for inference.
        Args:
            ds: DatasetInference."""

        val_loader = DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )
        return val_loader

    def run_token_generation(self, max_gen_tokens: int = 50):
        # TODO add docstring
        with tqdm(self.val_dl, unit="batch") as pbar:
            for x, paths in pbar:
                batch_size = x.shape[0]
                x = x.to(Config.device)
                pred_input = (
                    torch.ones((batch_size, 1))
                    .fill_(self.tokenizer.BOS)
                    .long()
                    .to(Config.device)
                )
                pred_probs = torch.ones((batch_size, 1))
                with torch.no_grad():
                    x_enc = self.model.encode_x(x)
                for gen_step in range(max_gen_tokens):
                    pbar.set_description(f"Generating token: {gen_step}")

                    with torch.no_grad():
                        y_pred = self.model.generate(x_enc, pred_input)
                    predicted_token = (
                        torch.softmax(y_pred, dim=-1).argmax(dim=-1).unsqueeze(dim=1)
                    )
                    probs = (
                        torch.max(F.softmax(y_pred, dim=-1), dim=-1)
                        .values.cpu()
                        .unsqueeze(dim=1)
                    )
                    pred_input = torch.cat([pred_input, predicted_token], dim=1)
                    pred_probs = torch.cat([pred_probs, probs], dim=1)
                decoded_tokens = self.tokenizer.decode_tokens_from_generation(
                    tokens=pred_input[:, 1:], probs=pred_probs[:, 1:]
                )
                for k, path in enumerate(paths):
                    self.predictions[path] = decoded_tokens[k]
                    print(f"Results for img: {path}")
                    pprint(decoded_tokens[k])

                torch.cuda.empty_cache()

    def inference(self):
        self.run_token_generation()

    def create_output_images(self):
        """Overlays the images with the predicted boxes."""
        for path, prediction in self.predictions.items():
            self.dataset.draw_patchwise_boundingboxes(
                Image.open(path), predictions=prediction
            ).save(f"{path.name.split('.')[0]}_pred.jpg")
