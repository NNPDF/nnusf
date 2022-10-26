# -*- coding: utf-8 -*-
import numpy as np


def kinematics_mapping(dataset, max_kin_value):
    """Rescale the input kinematic values (expect `x`) to be
    between -1 and 1.
    """
    scaled_inputs = []
    for index, kin_var in enumerate(dataset):
        num = kin_var - max_kin_value[index][0]
        den = max_kin_value[index][1] - max_kin_value[index][0]
        scaled_inputs.append(2 * (num / den) - 1)
    return scaled_inputs


def extract_extreme_values(datasets):
    """Store the maximum values of the given kinematics (x, Q2, A)
    into a list and use them to rescale the input kinematics.
    """
    data_kin = [data.kinematics for data in datasets.values()]
    data_kin = np.concatenate(data_kin, axis=0)
    extreme_values = np.stack([data_kin.min(axis=0),data_kin.max(axis=0)])
    return extreme_values


def apply_mapping_datasets(datasets, max_kin_value):
    for dataset in datasets.values():
        scaled = kinematics_mapping(dataset.kinematics.T, max_kin_value)
        dataset.kinematics = np.array(scaled).T


def rescale_inputs(datasets, method="extract_max_value"):
    function_call = globals()[method]
    max_kin_value = function_call(datasets)
    apply_mapping_datasets(datasets, max_kin_value)
