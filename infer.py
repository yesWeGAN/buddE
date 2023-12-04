#!/usr/bin/env python
import argparse
import importlib
import inference_engine

importlib.reload(inference_engine)
from config import Config
from inference_engine import DatasetInference, ModelInference
from tokenizer import PatchwiseTokenizer
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor
from model import ODModel
from utils import load_latest_checkpoint
import torch
import importlib
import inference_engine


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Process some integers.")

    # the actual arguments

    parser.add_argument(
        "-d",
        "--dir",
        help="Directory path where images to infer are.",
    )
    parser.add_argument("-p", "--probs", help="Minimum class probability.")
    parser.add_argument(
        "-c",
        "--create_imgs",
        action="store_true",
        help="Create images with bounding boxes.",
    )

    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
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
    ds = DatasetInference(root=inputs.dir, preprocessor=processor, tokenizer=tokenizr)
    # setup model
    modelp = ODModel(tokenizer=tokenizr)
    modelp.load_state_dict(latest_checkpoint["model_state_dict"])
    modelp.to("cuda:0")
    inference_pipeline = ModelInference(
        model=modelp,
        dataset=ds,
    )
    inference_pipeline.inference()
    if inputs.create_imgs:
        print("Creating output images...")
        inference_pipeline.create_output_images()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
