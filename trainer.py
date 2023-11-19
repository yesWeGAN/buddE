import torch
from torch.utils.data import Dataset, DataLoader
from dataset import DatasetODT
from typing import Callable
import numpy as np
import torch.nn.functional as F
import wandb
from utils import LOGGING
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint
from transformers import get_linear_schedule_with_warmup


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ModelTrainer:
    """This class handles the model training and absorbs boilerplate code."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: DatasetODT,
        config: dict,
        metric_callable: Callable = None,
        pad_token: int = None,
        start_epoch: int = 0,
        run_id: str = "dummy",
    ):
        self.lr = float(config["training"]["lr"])
        self.epochs = int(config["training"]["epochs"])
        self.batch_size = int(config["training"]["batch_size"])
        self.num_workers = int(config["training"]["num_workers"])
        self.weight_decay = float(config["training"]["weight_decay"])
        self.start_epoch = start_epoch

        self.metric = (
            metric_callable
            if metric_callable is not None
            else MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        )
        self.model = model

        self.train_dl, self.val_dl = self._setup_dl(ds=dataset, config=config)
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = (
            self._setup_lr_scheduler() if self.start_epoch == 0 else None
        )
        self.tokenizer = dataset.tokenizer

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
        self.run_id = run_id
        self.logger = None  # TODO: logger in utils for wandb

    def _setup_lr_scheduler(self):
        training_steps = self.epochs * (len(self.train_dl.dataset) // self.batch_size)
        warmup_steps = int(0.05 * training_steps)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=training_steps,
            num_warmup_steps=warmup_steps,
        )

    def _setup_dl(self, ds: DatasetODT, config: dict) -> tuple:
        """Takes a dataset and prepares DataLoaders for training and validation.
        Args:
            ds: DatasetODT."""
        train_split = DatasetODT(
            config=config,
            preprocessor=ds.preprocessor,
            tokenizer=ds.tokenizer,
            transforms=ds.transforms,
            split="train",
        )
        val_split = DatasetODT(
            config=config,
            preprocessor=ds.preprocessor,
            tokenizer=ds.tokenizer,
            transforms=ds.transforms,
            split="val",
        )
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )
        val_loader = DataLoader(
            dataset=val_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ds.collate_fn,
        )
        return train_loader, val_loader

    def _setup_optimizer(self):
        return torch.optim.Adam(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_validate_one_epoch(self, epoch: int) -> tuple:
        """Train one epoch and record losses / metrics.
        Args:
            epoch: Current epoch to train."""

        self.model = self.model.train()
        self.model = self.model.to(device)

        train_losses = []
        val_losses = []
        print(f"Training epoch: {epoch}")

        for x, y in self.train_dl:
            x = x.to(device)
            y = y.to(device)
            y_without_EOS = y[
                :, :-1
            ]  # the input to the model. No token to predict after EOS
            y_shifted_right = y[:, 1:]  # we do not assign loss for the SOS token
            # not an RNN, so we zero_grad in each step
            y_pred = self.model(x, y_without_EOS).permute(
                0, 2, 1
            )  # criterion expects logits in BATCH, VOCAB_SIZE, OTHER_DIMS*
            loss = self.criterion(y_pred, y_shifted_right)
            print(f"Train step loss: {loss.item():.5f}", end="\r")
            self.update_metric(y_pred, y_shifted_right)

            if LOGGING:
                wandb.log({"train_loss": loss.item()})
            train_losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.model = self.model.eval()

        for x, y in self.val_dl:
            x = x.to(device)
            y = y.to(device)
            y_without_EOS = y[
                :, :-1
            ]  # the input to the model. No token to predict after EOS
            y_shifted_right = y[:, 1:]  # we do not assign loss for the SOS token
            with torch.no_grad():
                y_pred = self.model(x, y_without_EOS).permute(0, 2, 1)
            loss = self.criterion(y_pred, y_shifted_right)
            print(f"Validation step loss: {loss.item():.5f}", end="\r")
            self.update_metric(y_pred, y_shifted_right)
            if LOGGING:
                wandb.log({"val_loss": loss.item()})
            val_losses.append(loss.item())
        torch.cuda.empty_cache()
        return train_losses, val_losses

    def train(self):
        """Trains the model through all epochs. Saves best checkpoints."""
        best_val_loss = float("inf")
        best_map = 0.0
        for epoch in range(self.start_epoch, self.epochs):
            _, val_loss = self.train_validate_one_epoch(epoch=epoch)

            avg_val_loss = np.mean(np.array(val_loss))
            if avg_val_loss < best_val_loss:
                print("New best average val loss!")
                best_val_loss = avg_val_loss

            results = self.metric.compute()
            pprint(results)
            self.metric.reset()
            if results["map"] > best_map:
                print("New best mAP! Saving model..")
                self.store_checkpoint(epoch=epoch)
                best_map = results["map"]

            if LOGGING:
                wandb.log({"mAP": results["map"]})

    def update_metric(self, pred, target) -> None:
        """Update the torchmetric metric with predictions and targets."""
        pred_decoded = self.tokenizer.decode_tokens(pred, return_scores=True)
        target_decoded = self.tokenizer.decode_tokens(target)
        self.metric.update(pred_decoded, target_decoded)

    def store_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "run_id": self.run_id,
            },
            f"checkpoint_epoch_{epoch}.pt",
        )
