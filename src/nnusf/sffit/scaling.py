# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import PchipInterpolator


def kinematics_mapping(dataset, map_from, map_to):
    scaled_inputs = []
    for index, kin_var in enumerate(dataset):
        # Scale only along the (Q2, A) directions
        if index != 0:
            scaler = PchipInterpolator(
                map_from[index],
                map_to[index],
                extrapolate=True,
            )
            input_scaling = scaler(kin_var)
        else:
            input_scaling = kin_var
        scaled_inputs.append(input_scaling)
    return scaled_inputs


def linear_rescaling(datasets):
    data_kin = [data.kinematics for data in datasets.values()]

    # Combined and sort each column of the kinematics (x, Q2, A)
    sorted_kin = np.sort(np.concatenate(data_kin, axis=0), axis=0)

    mapping_from, mapping_to = [], []
    for kin_var in sorted_kin.T:
        kin_unique, _ = np.unique(kin_var, return_counts=True)
        # Just rescales the inputs to be between 0 and 1
        scaling_target = kin_unique / kin_unique.max()

        mapping_from.append(kin_unique)
        mapping_to.append(scaling_target)

    return mapping_from, mapping_to


def apply_mapping_datasets(datasets, map_from, map_to):
    for dataset in datasets.values():
        scaled = kinematics_mapping(dataset.kinematics.T, map_from, map_to)
        dataset.kinematics = np.array(scaled).T


def rescale_inputs(datasets, method="linear_rescaling"):
    function_call = globals()[method]
    map_from, map_to = function_call(datasets)
    apply_mapping_datasets(datasets, map_from, map_to)
