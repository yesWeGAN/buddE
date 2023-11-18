#!/usr/bin/env python
import argparse
import sys
from pprint import pprint

import toml
from dataset import DatasetODT

from tokenizer import PatchwiseTokenizer
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from trainer import ModelTrainer
from model import ODModel
import wandb
from utils import LOGGING


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Process some integers.")

    # the actual arguments
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config.toml",
        help="Path to the config file.",
    )

    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
    config = toml.load(inputs.config_path)
    print("Starting training with args:")
    # pprint(config)

    if LOGGING:
        run = wandb.init(
            project="object-detection-transformer",
        )

        wandb.config = {"lr": float(config["training"]["lr"]),
                        "epochs" : int(config["training"]["epochs"]),
                        "batch_size" : int(config["training"]["batch_size"]),
                        "weight_decay" : float(config["training"]["weight_decay"]),
                        "pretrained_encoder": config["encoder"]["pretrained_encoder"],
                        "encoder_bottleneck": config["encoder"]["encoder_bottleneck"],
                        "num_decoder_layers": config["decoder"]["num_decoder_layers"],
                        "decoder_layer_dim": config["decoder"]["decoder_layer_dim"],
                        "num_heads": config["decoder"]["num_heads"],
                        "patch_size": config["transforms"]["patch_size"],}
    # setup tokenizer
    tokenizr = PatchwiseTokenizer(config=config)
    
    # setup the image processor
    processor = DeiTImageProcessor()

    # setup the dataset
    ds = DatasetODT(
        config=config,
        preprocessor=processor,
        tokenizer=tokenizr,
    )
    model = ODModel(config=config, tokenizer=tokenizr)
    coach = ModelTrainer(model=model, dataset=ds, config=config, pad_token=tokenizr.PAD)
    coach.train()


if __name__ == "__main__":
    main()
