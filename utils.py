from typing import Union
import json
import torch
from pathlib import Path
import os

def read_json_annotation(filepath: Union[str, Path]) -> dict:
    "Reads a json file from path and returns its content."
    with open(filepath, "r") as jsonin:
        return json.load(jsonin)
    
def load_latest_checkpoint(checkpoint_dir: Union[str, Path]=None) -> dict:
    """Loads the latest checkpoint in the given checkpoint dir."""
    try:
        filep = next(iter(sorted(Path(".").glob("*.pt"), key=os.path.getmtime, reverse=True)))
        print(f"Resuming from checkpoint: {filep.name}")
    except StopIteration:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}. Exiting.")
    return torch.load(filep)



    