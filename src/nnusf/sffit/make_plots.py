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
    parser.add_argument(
        "--output", default="output"
    )
    args = parser.parse_args()

    path_to_runcard = pathlib.Path(args.runcard)

    path_to_output_folder = pathlib.Path(args.output)

    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.FullLoader)

    import ipdb; ipdb.set_trace()
    for action in runcard_content["actions"]:
        func = getattr(plots, action)
        func(**runcard_content["actions"])


if __name__ == "__main__":
    main()
