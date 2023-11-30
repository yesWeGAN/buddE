#!/usr/bin/env python
import argparse
from pprint import pprint
from config import Config
from dataset import DatasetODT
import importlib
import model
import trainer
importlib.reload(model)
importlib.reload(trainer)

from tokenizer import PatchwiseTokenizer
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from trainer import ModelTrainer, ModelEvaluator
from model import ODModel
import wandb
from utils import load_latest_checkpoint


def main():
    # setup tokenizer
    tokenizr = PatchwiseTokenizer()

    # setup the image processor
    processor = DeiTImageProcessor(
        size={
            "height": Config.target_image_size,
            "width": Config.target_image_size,
        },
        do_center_crop=False,
    )
    latest_checkpoint = load_latest_checkpoint(".")
    # setup the dataset
    ds = DatasetODT(
        preprocessor=processor,
        tokenizer=tokenizr,
    )
    # setup model
    modelp = ODModel(tokenizer=tokenizr)
    modelp.load_state_dict(latest_checkpoint["model_state_dict"])
    modelp.to("cuda:0")
    evaluator = ModelEvaluator(
        model=modelp,
        dataset=ds,
    )
    evaluator.validate()
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":

    main()
