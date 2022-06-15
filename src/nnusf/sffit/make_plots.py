"""
Executable to make plots
"""

import argparse
import logging
import pathlib

import yaml

from . import plots

_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="sffit - fits sfs")
    parser.add_argument(
        "runcard",
    )
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    path_to_runcard = pathlib.Path(args.runcard)
    path_to_output_folder = pathlib.Path(args.output)

    if path_to_output_folder.exists():
        _logger.warning(f"{path_to_output_folder} already exists, overwriting content.")
    path_to_output_folder.mkdir(parents=True, exist_ok=True)

    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.SafeLoader)

    runcard_content["output"] = str(path_to_output_folder.absolute())

    for action in runcard_content["actions"]:
        func = getattr(plots, action)
        func(**runcard_content)
