# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import PchipInterpolator


def cumulative_rescaling(datasets, interpolation_points=None):
    data_kin = [data.kinematics for data in datasets.values()]

    # Combined and sort each column of the kinematics (x, Q2, A)
    sorted_kin = np.sort(np.concatenate(data_kin, axis=0), axis=0)

    # Define a combined dense target kinematic grids
    target_grids = np.linspace(
        start=-1.0,
        stop=1.0,
        endpoint=True,
        num=sorted_kin.shape[0],
    )

    scaler_funcs = []
    for kin_var in sorted_kin.T:
        kin_unique, kin_counts = np.unique(kin_var, return_counts=True)
        scaling_target = [
            target_grids[cumlsum - kin_counts[0]]
            for cumlsum in np.cumsum(kin_counts)
        ]
        if interpolation_points:
            kin_unique = kin_unique[0:len(kin_unique):int(len(kin_unique)/interpolation_points+1)]
            scaling_target = scaling_target[0:len(scaling_target):int(len(scaling_target)/interpolation_points+1)]
        interpolation_func = PchipInterpolator(kin_unique, scaling_target, extrapolate=True)
        scaler_funcs.append(interpolation_func)

    for dataset in datasets.values():
        scaled_inputs = []
        for index, kin_var in enumerate(dataset.kinematics.T):
            input_scaling = scaler_funcs[index](kin_var)
            scaled_inputs.append(input_scaling)
        dataset.kinematics = np.array(scaled_inputs).T


def rescale_inputs(datasets, interpolation_points, method="cumulative_rescaling"):
    function_call = globals()[method]
    function_call(datasets, interpolation_points)
