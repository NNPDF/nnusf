# -*- coding: utf-8 -*-
import numpy as np


def kinematics_mapping(dataset, map_from, map_to):
    scaled_inputs = []
    print(dataset.shape)
    exit()
    for index, kin_var in enumerate(dataset):
        # Scale only alon (Q2, A) directions
        if index != 0:
            input_scaling = np.interp(
                kin_var,
                map_from[index],
                map_to[index],
            )
        else:
            input_scaling = kin_var
        scaled_inputs.append(input_scaling)
    return scaled_inputs


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
        # TODO: Do not forget to remove the transformation below
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


def apply_mapping_datasets(datasets, map_from, map_to):
    for dataset in datasets.values():
        scaled = kinematics_mapping(dataset.kinematics.T, map_from, map_to)
        dataset.kinematics = np.array(scaled).T


def rescale_inputs(datasets, method="cumulative_rescaling"):
    function_call = globals()[method]
    map_from, map_to = function_call(datasets)
    apply_mapping_datasets(datasets, map_from, map_to)
