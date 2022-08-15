# -*- coding: utf-8 -*-
import numpy as np


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


def rescale_inputs(datasets, kls, esk, method=None):
    # kls, esk = cumulative_rescaling(datasets)
    for dataset in datasets.values():
        scaled_inputs = []
        for index, kin_var in enumerate(dataset.kinematics.T):
            input_scaling = np.interp(kin_var, kls[index], esk[index])
            scaled_inputs.append(input_scaling)
        dataset.kinematics = np.array(scaled_inputs).T
