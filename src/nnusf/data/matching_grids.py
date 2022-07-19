# -*- coding: utf-8 -*-
"""
Generate matching grids
"""

import itertools
import logging
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pineappl
from eko.interpolation import make_lambert_grid

from .. import utils
from ..theory.predictions import pdf_error
from .utils import (
    MAP_OBS_PID,
    build_obs_dict,
    construct_uncertainties,
    dump_info_file,
    write_to_csv,
)

_logger = logging.getLogger(__name__)

q2_min = 300
q2_max = 1e5

x_min = 1e-5

y_min = 0.2
y_max = 0.8


N_KINEMATC_GRID_FX = dict(x=50, Q2=100, y=1.0)
N_KINEMATC_GRID_XSEC = dict(x=30, Q2=50, y=4)
M_PROTON = 938.272013 * 0.001


def proton_boundary_conditions(
    pdf: str, obstype: str, destination: pathlib.Path
) -> None:
    """Generate the Yadism data (kinematicas & central values) as
    well as the the predictions for all replicas for A=1 used to
    impose the Boundary Condition.

    Parameters
    ----------
    pdf : str
        name of the PDF set to convolute with
    obstype : str
        bservable type: F2, F3, DXDYNUU, DXDYNUB
    destination : pathlib.Path
        destination to store the files
    """
    destination.mkdir(parents=True, exist_ok=True)
    _logger.info(f" Boundary condition grids destination : {destination}")

    obspid = MAP_OBS_PID[obstype]
    obsdic = build_obs_dict(obstype, [None], obspid)
    datapath = pathlib.Path(f"DATA_PROTONBC_{obspid}")

    q2_min = 1.65  # Redifine Q2min in case of A=1
    main(datapath, pdf, destination)
    dump_info_file(destination, "PROTONBC", [obsdic], 1, M_PROTON)


def main(grids: pathlib.Path, pdf: str, destination: pathlib.Path) -> None:
    """Generate the Yadism data (kinematicas & central values) as
    well as the the predictions for all replicas.

    Parameters
    ----------
    grids : pathlib.Path
        path to the pineappl.tar.gz grid
    pdf : str
        name of the PDF set to convolute with
    destination : pathlib.Path
        destination to store the files
    """
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        grid_name = grids.stem[6:-13]
        obs = grid_name.split("_")[-1]
        new_name = f"{grid_name}_MATCHING"

        # if grids.suffix == ".tar.gz":
        if str(grids).endswith(".tar.gz"):
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

        full_pred = []
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            grid = pineappl.grid.Grid.read(gpath)
            prediction = pdf_error(grid, pdf, x_grid, reshape=False)
            full_pred.append(prediction[0])
        pred = np.average(full_pred, axis=0)

        kinematics = {"x": [], "Q2": [], "y": []}
        err_list = []
        for x, q2, y in itertools.product(x_grid, q2_grid, y_grid):
            kinematics["x"].append(x)
            kinematics["Q2"].append(q2)
            kinematics["y"].append(y)
            err_list.append({"stat": 0.0, "syst": 0.0})

        kinematics_pd = pd.DataFrame(kinematics)
        # Select only predictions for Replicas_0 in data
        data_pd = pd.DataFrame({"data": pred[:, 0]})
        errors_pd = construct_uncertainties(err_list)

        # Dump the kinematics into CSV
        kinematics_folder = destination.joinpath("kinematics")
        kinematics_folder.mkdir(exist_ok=True)
        if is_xsec:
            write_to_csv(kinematics_folder, f"KIN_MATCHING_XSEC", kinematics_pd)
        else:
            write_to_csv(kinematics_folder, f"KIN_MATCHING_FX", kinematics_pd)

        # Dump the central (replica) data into CSV
        central_val_folder = destination.joinpath("data")
        central_val_folder.mkdir(exist_ok=True)
        write_to_csv(central_val_folder, f"DATA_{new_name}", data_pd)

        # Dump the dummy incertainties into CSV
        systypes_folder = destination.joinpath("uncertainties")
        systypes_folder.mkdir(exist_ok=True)
        write_to_csv(systypes_folder, f"UNC_{new_name}", errors_pd)

        # Dump the predictions for the REST of the replicas as NPY
        # NOTE: The following does no longer contain the REPLICA_0
        pred_folder = destination.joinpath("matching")
        pred_folder.mkdir(exist_ok=True)
        mat_dest = (pred_folder / f"MATCH_{new_name}").with_suffix(".npy")
        np.save(mat_dest, pred[:, 1:])

        msg = f"The matching/BC grid for {grid_name} are stored in "
        msg += f"'{destination.absolute().relative_to(pathlib.Path.cwd())}'"
        _logger.info(msg)
