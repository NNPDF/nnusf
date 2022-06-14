"""
Executable to perform the structure function fit
"""

import argparse
import logging
import pathlib
import yaml
import shutil

import nnusf.sffit.load_data as load_data
from rich.logging import RichHandler
from nnusf.sffit.model_gen import generate_models
from nnusf.sffit.train_model import perform_fit

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="sffit - fits sfs")
    parser.add_argument("runcard")
    parser.add_argument("replica", type=int)
    args = parser.parse_args()

    path_to_runcard = pathlib.Path(args.runcard)

    # Create a folder for the replica
    path_to_fit_folder = pathlib.Path(path_to_runcard.stem) / f"replica_{args.replica}"
    if path_to_fit_folder.exists():
        log.warning(f"{path_to_fit_folder} already exists, overwriting content.")
    path_to_fit_folder.mkdir(parents=True, exist_ok=True)

    # copy runcard to the fit folder
    shutil.copy(path_to_runcard, path_to_fit_folder.parent / "runcard.yml")

    with open(path_to_runcard) as file:
        runcard_content = yaml.load(file, Loader=yaml.FullLoader)

    experiments_dict = runcard_content["experiments"]
    data_info = load_data.load_experimental_data(experiments_dict)

    # create pseudodata and add it to the data_info object
    load_data.add_pseudodata(data_info)

    fit_dict = generate_models(data_info, **runcard_content["fit_parameters"])

    # Compile the training and validationa nd perform the fit
    perform_fit(fit_dict, data_info, **runcard_content["fit_parameters"])

    # Store the models in the relevant replica subfolders
    # saved_model = tf.keras.Model(
    #     inputs=tr_model.inputs,
    #     outputs=tr_model.get_layer("SF_output").output
    # )
    # saved_model.save(path_to_fit_folder / "model")


if __name__ == "__main__":
    main()
