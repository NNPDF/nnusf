"""
Executable to perform the structure function fit
"""

import argparse
import pathlib
import yaml
from load_data import load_experimental_data, make_pseudodata


def main():
    parser = argparse.ArgumentParser(description="sffit - fits sfs")
    parser.add_argument(
        "runcard",
    )
    args = parser.parse_args()

    path_to_runcard = pathlib.Path(args.runcard)
    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.FullLoader)

    experiments_dict = runcard_content["experiments"]
    experimental_data = load_experimental_data(experiments_dict)

    if runcard_content["pseudodataseed"]:
        pseudo_data = make_pseudodata(experimental_data)


    # Load data

    # Build model

    # Do fit


if __name__ == "__main__":
    main()