from typing import Any, Union, Callable
import json

from pathlib import Path

LOGGING = True
#LOGGING = False

def read_json_annotation(filepath: Union[str, Path]) -> dict:
    "Reads a json file from path and returns its content."
    with open(filepath, "r") as jsonin:
        return json.load(jsonin)
    