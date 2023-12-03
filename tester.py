import torch
from torch.utils.data import DataLoader
from dataset import DatasetODT
from typing import Callable
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm
from config import Config
from tokenizer import PatchwiseTokenizer
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:
    """This class handles the model evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: DatasetODT,
        metric_callable: Callable = None,
        pad_token: int = None,
    ):
        self.batch_size = Config.validation_batch_size
        self.num_workers = Config.num_workers

        self.metric = (
            metric_callable
            if metric_callable is not None
            else MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        )

        self.model = model.eval()
        self.val_dl = self._setup_dl(ds=dataset)
        self.tokenizer: PatchwiseTokenizer = dataset.tokenizer

    def _setup_dl(self, ds: DatasetODT) -> tuple:
        """Takes a dataset and prepares DataLoaders for validation.
        Args:
            ds: DatasetODT."""

        val_split = DatasetODT(
            preprocessor=ds.preprocessor,
            tokenizer=ds.tokenizer,
            transforms=ds.transforms,
            split="val",
        )

        val_loader = DataLoader(
            dataset=val_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )
        return val_loader

    def run_token_generation(self, max_gen_tokens: int = 50):
        # TODO add docstring
        with tqdm(self.val_dl, unit="batch") as pbar:
            for x, y in pbar:
                batch_size = x.shape[0]
                x = x.to(device)
                y_shifted_right = y[:, 1:]
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
                self.update_metric(
                    pred_input[:, 1:], y_shifted_right, pred_probs[:, 1:]
                )
                torch.cuda.empty_cache()

    def update_metric(self, pred, target, probs) -> None:
        """Update the torchmetric metric with predictions and targets.
        TODO: add type hints"""
        pred_decoded = self.tokenizer.decode_tokens_from_generation(
            tokens=pred, probs=probs
        )
        target_decoded = self.tokenizer.decode_tokens(target)
        self.metric.update(pred_decoded, target_decoded)

    def validate(self):
        self.run_token_generation()
        pprint(self.metric.compute())
