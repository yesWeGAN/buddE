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
from utils import LOGGING, load_latest_checkpoint


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
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )

    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
    config = toml.load(inputs.config_path)
    print("Starting training with args:")
    pprint(config)

    if LOGGING:
        wandbconfig = {
            "lr": float(config["training"]["lr"]),
            "epochs": int(config["training"]["epochs"]),
            "batch_size": int(config["training"]["batch_size"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "pretrained_encoder": config["encoder"]["pretrained_encoder"],
            "encoder_bottleneck": config["encoder"]["encoder_bottleneck"],
            "num_decoder_layers": config["decoder"]["num_decoder_layers"],
            "decoder_layer_dim": config["decoder"]["decoder_layer_dim"],
            "num_heads": config["decoder"]["num_heads"],
            "patch_size": config["transforms"]["patch_size"],
        }

        wandb.init(project="object-detection-transformer", config=wandbconfig)
        run_id = wandb.run.id

    if inputs.resume:
        latest_checkpoint = load_latest_checkpoint(".")

    # setup tokenizer
    tokenizr = PatchwiseTokenizer(config=config)

    # setup the image processor
    processor = DeiTImageProcessor(
        size={
            "height": config["transforms"]["target_image_size"],
            "width": config["transforms"]["target_image_size"],
        },
        do_center_crop=False,
    )

    # setup the dataset
    ds = DatasetODT(
        config=config,
        preprocessor=processor,
        tokenizer=tokenizr,
    )
    # setup model
    model = ODModel(config=config, tokenizer=tokenizr)

    if inputs.resume:
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        model.to("cuda:0")
        coach = ModelTrainer(
            model=model,
            dataset=ds,
            config=config,
            pad_token=tokenizr.PAD,
            start_epoch=latest_checkpoint["epoch"] + 1,
        )
        coach.optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
    else:
        coach = ModelTrainer(
            model=model, dataset=ds, config=config, pad_token=tokenizr.PAD
        )

    coach.train()


if __name__ == "__main__":
    main()
