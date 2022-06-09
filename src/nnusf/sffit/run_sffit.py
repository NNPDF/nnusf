"""
Executable to perform the structure function fit
"""

import argparse
import pathlib

import yaml

import load_data
from model_gen import generate_models
from train_model import perform_fit

import logging
import tensorflow as tf



logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="sffit - fits sfs")
    parser.add_argument(
        "runcard",
    )
    parser.add_argument(
        "replica", type=int,
    )
    args = parser.parse_args()

    path_to_runcard = pathlib.Path(args.runcard)

    path_to_fit_folder = pathlib.Path(path_to_runcard.stem) / f"replica_{args.replica}"
    if path_to_fit_folder.exists():
        log.warning(f"{path_to_fit_folder} already exists, overwriting content.")
    path_to_fit_folder.mkdir(parents=True, exist_ok=True)

    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.FullLoader)

    experiments_dict = runcard_content["experiments"]
    data_info = load_data.load_experimental_data(experiments_dict)

    if runcard_content["pseudodataseed"]:
        load_data.add_pseudodata(data_info)

    load_data.add_tr_filter_mask(data_info, runcard_content["trvlseed"])

    tr_model, vl_model = generate_models(
        data_info, **runcard_content["fit_parameters"]
    )

    perform_fit(
        tr_model, vl_model, data_info, **runcard_content["fit_parameters"]
    )

    saved_model = tf.keras.Model(inputs=tr_model.inputs, outputs=tr_model.get_layer("SF_output").output)
    saved_model.save(path_to_fit_folder / "model")

if __name__ == "__main__":
    main()
