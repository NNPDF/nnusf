"""
Executable to make plots
"""

import argparse
import pathlib

import yaml

import logging
import tensorflow as tf

import plots

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
        log.warning(
            f"{path_to_output_folder} already exists, overwriting content."
        )
    path_to_output_folder.mkdir(parents=True, exist_ok=True)

    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.FullLoader)

    runcard_content["output"] = str(path_to_output_folder.absolute())

    for action in runcard_content["actions"]:
        func = getattr(plots, action)
        func(**runcard_content)


if __name__ == "__main__":
    main()
