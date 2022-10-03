# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import PchipInterpolator

from .utils import select_subset_q2points


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


def cumulative_rescaling(datasets, q2points=None, kincuts=None):
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

    mapping_from, mapping_to = [], []
    for index, kin_var in enumerate(sorted_kin.T):
        kin_unique, kin_counts = np.unique(kin_var, return_counts=True)
        scaling_target = [
            target_grids[cumlsum - kin_counts[0]]
            for cumlsum in np.cumsum(kin_counts)
        ]

        # If necessary, select smaller points in Q2
        if q2points is not None and index == 1:
            kin_unique, scaling_target = select_subset_q2points(
                kin_unique,
                scaling_target,
                q2points,
                kincuts,
            )

        mapping_from.append(kin_unique)
        mapping_to.append(scaling_target)

    return mapping_from, mapping_to


def apply_mapping_datasets(datasets, map_from, map_to):
    for dataset in datasets.values():
        scaled = kinematics_mapping(dataset.kinematics.T, map_from, map_to)
        dataset.kinematics = np.array(scaled).T


def rescale_inputs(
    datasets, q2points=None, kincuts=None, method="cumulative_rescaling"
):
    function_call = globals()[method]
    map_from, map_to = function_call(datasets, q2points, kincuts)
    apply_mapping_datasets(datasets, map_from, map_to)
