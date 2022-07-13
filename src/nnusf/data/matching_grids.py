# -*- coding: utf-8 -*-
"""
Generate matching grids
"""

import itertools
import logging
import pathlib
import shutil

import numpy as np
import pandas as pd
from eko.interpolation import make_lambert_grid

from .utils import construct_uncertainties, write_to_csv

_logger = logging.getLogger(__name__)

q2_min = 1.0
q2_max = 1e5

x_min = 1e-7

y_min = 0.2
y_max = 0.8


def main(destination: pathlib.Path, datapaths: list[pathlib.Path]):
    destination.mkdir(parents=True, exist_ok=True)
    _logger.info(f" Matching grids : {destination}")

    _logger.info("Saving coefficients:")
    for dataset in datapaths:
        data_name = dataset.stem.strip("DATA_")
        obs = data_name.split("_")[-1]
        new_name = f"MATCHING-{data_name}"

        is_xsec = "DXDY" in obs or "FW" in obs
        if is_xsec:
            n_xgrid = 30
            n_q2grid = 200
            n_ygrid = 5
            y_grid = np.linspace(y_min, y_max, n_ygrid)
        else:
            n_xgrid = 50
            n_q2grid = 400
            n_ygrid = 1.0
            y_grid = [0.0]

        _logger.info(f"Saving matching grids for {data_name} in {new_name}")

        x_grid = make_lambert_grid(n_xgrid, x_min)
        q2_grid = np.geomspace(q2_min, q2_max, n_q2grid)
        n_points = int(n_q2grid * n_ygrid * n_xgrid)

        kinematics = {"x": [], "Q2": [], "y": []}
        err_list = []
        for x, q2, y in itertools.product(x_grid, q2_grid, y_grid):
            kinematics["x"].append(x)
            kinematics["Q2"].append(q2)
            kinematics["y"].append(y)
            err_list.append({"stat": 0.0, "syst": 0.0})

        kinematics_pd = pd.DataFrame(kinematics)
        data_pd = pd.DataFrame({"data": np.zeros((n_points))})
        errors_pd = construct_uncertainties(err_list)

        kinematics_folder = destination.joinpath("kinematics")
        kinematics_folder.mkdir(exist_ok=True)
        if is_xsec:
            write_to_csv(kinematics_folder, f"KIN_MATCHING_XSEC", kinematics_pd)
        else:
            write_to_csv(kinematics_folder, f"KIN_MATCHING_FX", kinematics_pd)

        central_val_folder = destination.joinpath("data")
        central_val_folder.mkdir(exist_ok=True)
        write_to_csv(central_val_folder, f"DATA_{new_name}", data_pd)

        systypes_folder = destination.joinpath("uncertainties")
        systypes_folder.mkdir(exist_ok=True)
        write_to_csv(systypes_folder, f"UNC_{new_name}", errors_pd)
