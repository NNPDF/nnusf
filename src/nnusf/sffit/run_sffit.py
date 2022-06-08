"""
Executable to perform the structure function fit
"""

import argparse
import pathlib
import yaml
import load_data
import numpy as np
from model_gen import generate_models


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
    data_info = load_data.load_experimental_data(experiments_dict)

    if runcard_content["pseudodataseed"]:
        load_data.add_pseudodata(data_info)

    load_data.add_tr_filter_mask(data_info, runcard_content['trvlseed'])

    tr_model, vl_model = generate_models(data_info, **runcard_content["fit_parameters"])

    import ipdb; ipdb.set_trace()

    # Build model

    # Do fit


if __name__ == "__main__":
    main()