from nnusf.data.loader import Loader
import pathlib
import numpy as np
from dataclasses import dataclass

path_to_commondata = pathlib.Path(__file__).parents[3].joinpath("commondata")
path_to_theory = pathlib.Path(__file__).parents[3].joinpath("theory")

@dataclass
class Pseudodata:
    "Class containing data needed to fit an experimental dataset"
    name: str
    tr_mask: np.ndarray
    tr_kinematics_array: np.ndarray
    tr_invcovmat: np.ndarray
    tr_coefficients: np.ndarray
    tr_central_values: np.ndarray
    vl_kinematics_array: np.ndarray
    vl_invcovmat: np.ndarray
    vl_coefficients: np.ndarray
    vl_central_values: np.ndarray


def load_experimental_data(experiment_list):
    "returns a dictionary with dataset names as keys and data as value"
    experimental_data = {}
    for experiment in experiment_list:
        data = Loader(path_to_commondata, path_to_theory, experiment['dataset'])
        experimental_data[experiment['dataset']] = data
    return experimental_data


def make_pseudodata(experimental_data_list):
    pseudodata_dict = {}
    for dataset_name, experimental_data in zip(experimental_data_list.keys(), experimental_data_list.values()):
        cholesky = np.linalg.cholesky(experimental_data.covmat)
        random_samples = np.random.randn(experimental_data.covmat.shape[0])
        pseudodata = experimental_data.central_values + random_samples @ cholesky
        pseudodata_dict[dataset_name] = pseudodata
    return pseudodata_dict

def make_tr_vl_objects():
    pass
