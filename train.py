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

# see also code.data_collection.bing_collect for **kwargs implementation


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
    coach = ModelTrainer(model=model, dataset=ds, config=config)
    coach.train()


if __name__ == "__main__":
    main()
