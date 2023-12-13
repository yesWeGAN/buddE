#!/usr/bin/env python
import argparse

from torchmetrics.detection import MeanAveragePrecision
from transformers.models.deit.feature_extraction_deit import DeiTImageProcessor

from config import Config
from dataset import DatasetODT
from model import ODModel
from tester import ModelEvaluator
from tokenizer import PatchwiseTokenizer
from utils import load_latest_checkpoint


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Process some integers.")

    # the actual arguments

    parser.add_argument(
        "-c",
        "--class_metrics",
        action="store_true",
        help="Print metrics per class.",
    )

    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
    class_metrics = True if inputs.class_metrics else False
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
    latest_checkpoint = load_latest_checkpoint(Config.checkpoints_dir)
    # setup the dataset
    ds = DatasetODT(
        preprocessor=processor,
        tokenizer=tokenizr,
    )
    # setup model
    modelp = ODModel(tokenizer=tokenizr)
    modelp.load_state_dict(latest_checkpoint["model_state_dict"])
    modelp.to("cuda:0")
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=class_metrics
    )
    evaluator = ModelEvaluator(model=modelp, dataset=ds, metric_callable=metric)
    evaluator.validate()
    import torch

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
