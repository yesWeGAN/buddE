import argparse
import os
from pathlib import Path
import sys
import toml

parser = argparse.ArgumentParser()
parser.add_argument("config_fp", type="str")




def main():
    parsed_args = parser.parse_args()
    config = toml.load(parsed_args.config_fp)
    