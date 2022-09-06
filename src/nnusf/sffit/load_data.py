# -*- coding: utf-8 -*-
import pathlib
import random
from typing import Optional

import numpy as np

from ..data.loader import Loader

path_to_commondata = pathlib.Path(__file__).parents[3].joinpath("commondata")
path_to_coefficients = (
    pathlib.Path(__file__).parents[3].joinpath("coefficients")
)


def load_experimental_data(experiment_list, kincuts: dict = {}):
    "returns a dictionary with dataset names as keys and data as value"
    experimental_data = {}
    for experiment in experiment_list:
        data = Loader(
            experiment["dataset"],
            path_to_commondata=path_to_commondata,
            path_to_coefficients=path_to_coefficients,
            kincuts=kincuts,
        )
        data.tr_frac = experiment["frac"]
        experimental_data[experiment["dataset"]] = data
    return experimental_data


def add_pseudodata(experimental_datasets, shift=True):
    """If `shift=False` no pseudodata is generated and real data is used
    instead. This is only relevant for debugging purposes.
    """
    for dataset in experimental_datasets.values():
        cholesky = np.linalg.cholesky(dataset.covmat)
        random_samples = np.random.randn(dataset.n_data)
        shift_data = random_samples @ cholesky if shift else 0
        pseudodata = dataset.central_values + shift_data
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
