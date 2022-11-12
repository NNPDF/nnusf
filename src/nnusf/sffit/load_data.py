# -*- coding: utf-8 -*-
import copy
import logging
import pathlib
import random
from typing import Optional

import numpy as np

from ..data.loader import Loader
from .scaling import rescale_inputs

_logger = logging.getLogger(__name__)

curr_path = pathlib.Path(__file__)
path_to_commondata = curr_path.parents[3].joinpath("commondata")
path_to_coefficients = curr_path.parents[3].joinpath("coefficients")


def construct_expdata_instance(experiment_list, kincuts, verbose=True):
    """Collect all the dataset instances into a dictionary.

    Parameters:
    -----------
    experiment_list: dict
        contains the information on a given dataset including name
        and training fraction
    kincuts: dict
        contains the information on the cuts to be applied
    verbose: bool=True
        print out log outputs from data.loader

    Returns:
    --------
    dict:
        dictionary containing the dataset instances with dataset
        name as key
    """
    experimental_data = {}
    for experiment in experiment_list:
        data = Loader(
            experiment["dataset"],
            path_to_commondata=path_to_commondata,
            path_to_coefficients=path_to_coefficients,
            kincuts=kincuts,
            verbose=verbose,
        )
        data.tr_frac = experiment["frac"]
        experimental_data[experiment["dataset"]] = data
    return experimental_data


def load_experimental_data(
    experiment_list,
    input_scaling: Optional[bool] = None,
    kincuts: dict = {},
    verbose: bool = True,
):
    """Calls to `construct_expdata_instance` to construct the dataset
    instances and apply input scaling if needed.

    Parameters:
    -----------
    experiment_list: dict
        contains the infomration on a given dataset including name
        and training fraction
    input_scaling: bool
        choose to scale or not the kinematic inputs
    kincuts: dict
        contains the information on the cuts to be applied
    verbose:
        print out log outputs from data.loader

    Returns:
    --------
    tuple(dict, dict):
        original and scaled dataset specs
    """
    experimental_data = construct_expdata_instance(
        experiment_list,
        kincuts,
        verbose=verbose,
    )
    raw_experimental_data = copy.deepcopy(experimental_data)

    # Perform Input Scaling if required
    if input_scaling:
        _logger.info("Input kinematics are being scaled.")
        rescale_inputs(experimental_data)
    return raw_experimental_data, experimental_data


def add_pseudodata(experimental_datasets, shift=True):
    """Add fluctuations to the experimental datasets.

    Parameters:
    -----------
    experimental_datasets: dict
        contains the information on alll the datasets
    shift: bool
        If `shift=False` no pseudodata is generated and real data is
        used instead. This is only relevant for debugging purposes.
    """
    for dataset in experimental_datasets.values():
        cholesky = np.linalg.cholesky(dataset.covmat)
        random_samples = np.random.randn(dataset.n_data)
        shift_data = cholesky @ random_samples if shift else 0
        pseudodata = dataset.central_values + shift_data
        dataset.pseudodata = pseudodata


def add_tr_filter_mask(experimental_datasets):
    """Add filter masks to the dataset instances.

    Parameters:
    -----------
    experimental_datasets: dict
        contains the information on all the datasets
    """
    for dataset in experimental_datasets.values():
        rnd_sample = random.sample(
            range(dataset.n_data),
            int(dataset.tr_frac * dataset.n_data),
        )
        tr_indices = np.array(rnd_sample, dtype=int)
        tr_filter = np.zeros(dataset.n_data, dtype=bool)
        tr_filter[tr_indices] = True
        dataset.tr_filter = tr_filter


def cumulative_rescaling(datasets):
    data_kin = [data.kinematics for data in datasets.values()]

    # Combined and sort each column of the kinematics (x, Q2, A)
    sorted_kin = np.sort(np.concatenate(data_kin, axis=0), axis=0)

    # Define a combined dense target kinematic grids
    target_grids = np.linspace(
        start=0,
        stop=1.0,
        endpoint=True,
        num=sorted_kin.shape[0],
    )

    equally_spaced_kinematics = []
    kin_linear_reference = []
    for kin_var in sorted_kin.T:
        kin_unique, kin_counts = np.unique(kin_var, return_counts=True)
        scaling_target = [
            target_grids[cumlsum - kin_counts[0]]
            for cumlsum in np.cumsum(kin_counts)
        ]
        kin_linear_spaced = np.linspace(
            kin_var[0],
            kin_var[-1],
            num=int(2 * kin_var.size),
        )
        equally_spaced_kinematics.append(
            np.interp(kin_linear_spaced, kin_unique, scaling_target)
        )
        kin_linear_reference.append(kin_linear_spaced)
    return kin_linear_reference, equally_spaced_kinematics
