#!/usr/bin/env python
import argparse
from pprint import pprint
from config import Config
from dataset import DatasetODT

from tokenizer import PatchwiseTokenizer
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from trainer import ModelTrainer
from model import ODModel
import wandb
from utils import load_latest_checkpoint


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Process some integers.")

    # the actual arguments

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
    wandbconfig = {
        "dataset": Config.dataset,
        "lr": Config.lr,
        "epochs": Config.epochs,
        "batch_size": Config.batch_size,
        "weight_decay": Config.weight_decay,
        "pretrained_encoder": Config.pretrained_encoder,
        "encoder_bottleneck": Config.encoder_bottleneck,
        "num_decoder_layers": Config.num_decoder_layers,
        "decoder_layer_dim": Config.decoder_layer_dim,
        "num_heads": Config.num_heads,
        "patch_size": Config.patch_size,
        "dropout": Config.dropout,
    }
    print("Starting training with args:")
    pprint(wandbconfig)

    if inputs.resume:
        latest_checkpoint = load_latest_checkpoint(".")

    if Config.logging:
        if inputs.resume:
            wandb.init(
                project="object-detection-transformer",
                config=wandbconfig,
                resume="allow",
                run_id=latest_checkpoint["run_id"],
            )
        else:
            wandb.init(project="object-detection-transformer", config=wandbconfig)
        run_id = wandb.run.id

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

    # setup the dataset
    ds = DatasetODT(
        preprocessor=processor,
        tokenizer=tokenizr,
    )
    # setup model
    model = ODModel(tokenizer=tokenizr)

    if inputs.resume:
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        model.to("cuda:0")
        coach = ModelTrainer(
            model=model,
            dataset=ds,
            pad_token=tokenizr.PAD,
            start_epoch=latest_checkpoint["epoch"] + 1,
            run_id=run_id,
        )
        coach.optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
    else:
        coach = ModelTrainer(model=model, dataset=ds, pad_token=tokenizr.PAD, run_id=run_id)

    coach.train()


if __name__ == "__main__":
    main()
