# -*- coding: utf-8 -*-
import numpy as np


def kinematics_mapping(dataset, max_kin_value):
    """Rescale the input kinematic values (expect `x`) to be
    between 0 and 1.
    """
    scaled_inputs = []
    for index, kin_var in enumerate(dataset):
        if index != 0:  # Scale only along (Q2, A) direction
            input_scaling = kin_var / max_kin_value[index]
        else:
            input_scaling = kin_var
        scaled_inputs.append(input_scaling)
    return scaled_inputs


def extract_extreme_values(datasets):
    """Store the maximum values of the given kinematics (x, Q2, A)
    into a list and use them to rescale the input kinematics.

    Parameters:
    -----------
    datasets: dict
        contains the dataset specs

    Returns:
    --------
    np.ndarray:
        maximum and minimum values of each of the input kinematics
    """
    data_kin = [data.kinematics for data in datasets.values()]
    data_kin = np.concatenate(data_kin, axis=0)
    return data_kin.max(axis=0)


def apply_mapping_datasets(datasets, max_kin_value):
    """Apply the rescaling to all the datasets.

    Parameters:
    -----------
    datasets: dict
        contains the dataset specs
    max_kin_value: np.ndarray
        maximum and minimum values of each of the input kinematics
    """
    for dataset in datasets.values():
        scaled = kinematics_mapping(dataset.kinematics.T, max_kin_value)
        dataset.kinematics = np.array(scaled).T


def rescale_inputs(datasets):
    """Apply the rescaling to all the datasets.

    Parameters:
    -----------
    datasets: dict
        contains the dataset specs
    """
    max_kin_value = extract_extreme_values(datasets)
    apply_mapping_datasets(datasets, max_kin_value)
