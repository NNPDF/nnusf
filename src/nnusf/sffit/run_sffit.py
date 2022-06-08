"""
Executable to perform the structure function fit
"""

import argparse
import pathlib
import yaml

def main():
    parser = argparse.ArgumentParser(description="sffit - fits sfs")
    parser.add_argument(
        "runcard",
    )
    args = parser.parse_args()

    path_to_runcard = pathlib.Path(args.runcard)
    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.FullLoader)

    # Load data

    # Build model

    # Do fit

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()