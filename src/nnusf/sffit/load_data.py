import pathlib
import random

import numpy as np
from nnusf.data.loader import Loader

path_to_commondata = pathlib.Path(__file__).parents[3].joinpath("commondata")
path_to_coefficients = (
    pathlib.Path(__file__).parents[3].joinpath("coefficients")
)


def load_experimental_data(experiment_list):
    "returns a dictionary with dataset names as keys and data as value"
    experimental_data = {}
    for experiment in experiment_list:
        data = Loader(
            experiment["dataset"], path_to_commondata, path_to_coefficients
        )
        data.tr_frac = experiment["frac"]
        experimental_data[experiment["dataset"]] = data
    return experimental_data


def add_pseudodata(experimental_datasets):
    for dataset in experimental_datasets.values():
        cholesky = np.linalg.cholesky(dataset.covmat)
        random_samples = np.random.randn(dataset.n_data)
        pseudodata = dataset.central_values + random_samples @ cholesky
        dataset.pseudodata = pseudodata


def add_tr_filter_mask(experimental_datasets, trvlseed=None):
    np.random.seed(seed=trvlseed)
    for dataset in experimental_datasets.values():
        tr_indices = np.array(
            random.sample(
                range(dataset.n_data), int(dataset.tr_frac * dataset.n_data)
            ),
            dtype=int,
        )
        tr_filter = np.zeros(dataset.n_data, dtype=bool)
        tr_filter[tr_indices] = True
        dataset.tr_filter = tr_filter
