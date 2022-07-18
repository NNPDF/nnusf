# -*- coding: utf-8 -*-
Generate matching grids
"""
"""

import os
import itertools
import pineappl
import logging
import pathlib

import numpy as np
import pandas as pd
import tempfile
from eko.interpolation import make_lambert_grid

from .utils import (
    build_obs_dict,
    construct_uncertainties,
    dump_info_file,
    write_to_csv,
)
from .. import utils
from ..theory.predictions import pdf_error, theory_error

_logger = logging.getLogger(__name__)

q2_min = 300 # PBC: Q2min: 1.65 GeV2
q2_max = 1e5

x_min = 1e-5

y_min = 0.2
y_max = 0.8


N_KINEMATC_GRID_FX = dict(x=50, Q2=200, y=1.0)
N_KINEMATC_GRID_XSEC = dict(x=30, Q2=50, y=4)
M_PROTON = 938.272013 * 0.001


def proton_boundary_conditions(destination: pathlib.Path, pdf:str):
    destination.mkdir(parents=True, exist_ok=True)
    _logger.info(f" Boundary condition grids destination : {destination}")

    datapaths = []
    obs_list = [
        build_obs_dict("F2", [None], 0),
        build_obs_dict("F3", [None], 0),
        build_obs_dict("DXDYNUU", [None], 14),
        build_obs_dict("DXDYNUB", [None], -14),
    ]
    for obs in obs_list:
        fx = obs["type"]
        datapaths.append(pathlib.Path(f"DATA_PROTONBC_{fx}"))

    main(destination, datapaths, pdf)

    dump_info_file(destination, "PROTONBC", obs_list, 1, M_PROTON)


def main(destination: pathlib.Path, grids: pathlib.Path, pdf: str):

    destination.mkdir(parents=True, exist_ok=True)

    grids = grids[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        grid_name = grids.stem.strip("grids-")
        obs = grid_name.split("_")[-1]
        new_name = f"{grid_name}_MATCHING"

        if grids.suffix == ".tar":
            utils.extract_tar(grids, tmpdir)
            grids = tmpdir / "grids"

        is_xsec = "DXDY" in obs or "FW" in obs
        if is_xsec:
            n_xgrid = N_KINEMATC_GRID_XSEC["x"]
            n_q2grid = N_KINEMATC_GRID_XSEC["Q2"]
            n_ygrid = N_KINEMATC_GRID_XSEC["y"]
            y_grid = np.linspace(y_min, y_max, n_ygrid)
        else:
            n_xgrid = N_KINEMATC_GRID_FX["x"]
            n_q2grid = N_KINEMATC_GRID_FX["Q2"]
            n_ygrid = N_KINEMATC_GRID_FX["y"]
            y_grid = [0.0]

        x_grid = make_lambert_grid(n_xgrid, x_min)
        q2_grid = np.linspace(q2_min, q2_max, int(n_q2grid))
        n_points = int(n_q2grid * n_ygrid * n_xgrid)

        full_pred = []
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            grid = pineappl.grid.Grid.read(gpath)
            pred, central, bulk, err_source = pdf_error(
                    grid, pdf, x_grid, reshape=False
            )
            full_pred.append(pred)
        pred = np.average(full_pred, axis=0)
        __import__('pdb').set_trace()

        kinematics = {"x": [], "Q2": [], "y": []}
        err_list = []
        for x, q2, y in itertools.product(x_grid, q2_grid, y_grid):
            kinematics["x"].append(x)
            kinematics["Q2"].append(q2)
            kinematics["y"].append(y)
            err_list.append({"stat": 0.0, "syst": 0.0})

        kinematics_pd = pd.DataFrame(kinematics)
        data_pd = pd.DataFrame({"data": np.zeros(n_points)})
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

        msg = f"The matching grid for {grid_name} are stored in "
        msg += f"'{destination.absolute().relative_to(pathlib.Path.cwd())}'"
        _logger.info(msg)
    os.removedirs(tmpdir)
