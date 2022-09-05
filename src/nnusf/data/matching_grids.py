# -*- coding: utf-8 -*-
"""
Generate matching grids
"""

import itertools
import logging
import pathlib
import tempfile
from typing import Optional, Tuple

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

KIN_DESC = {
    "x_min": 1e-2,
    "y_min": 0.2,
    "y_max": 0.8,
    "q2_min": 25,
    "q2_max": 400,
}


N_KINEMATC_GRID_FX = dict(x=8, Q2=25, y=1)
N_KINEMATC_GRID_XSEC = dict(x=8, Q2=25, y=2)
M_PROTON = 938.272013 * 0.001


def proton_boundary_conditions(
    destination: pathlib.Path,
    grids: Optional[list[pathlib.Path]] = None,
    pdf: Optional[str] = None,
) -> None:
    """Generate the Yadism data (kinematics & central values) as
    well as the the predictions for all replicas for A=1 used to
    impose the Boundary Condition.
    If no grids are provided, central values will be empty,
    and oly kinematics tables will be filled.

    Parameters
    ----------
    grids : list[pathlib.Path]
        list of paths to the pineappl.tar.gz grids
    pdf : str
        name of the PDF set to convolute with
    destination : pathlib.Path
        destination to store the files
    """
    destination.mkdir(parents=True, exist_ok=True)

    # Set Q2min for BC datasets
    KIN_DESC["q2_min"] = 1.65
    if grids is not None:
        obsdic_list = []
        for grid in grids:
            grid_name = grid.stem[6:-13]
            _logger.info(f"Generating BC data for '{grid_name}'.")

            obstype = grid_name.split("_")[-1]
            obspid = MAP_OBS_PID[obstype]
            obsdic = build_obs_dict(obstype, [None], obspid)
            obsdic_list.append(obsdic)
            main(
                grid, pdf, destination, kin=KIN_DESC, match_type="KIN_PROTONBC"
            )
    else:
        datapaths = []
        obsdic_list = [
            build_obs_dict("DXDYNUB", [None], -14),
            build_obs_dict("DXDYNUU", [None], 14),
            build_obs_dict("F2", [None], 0),
            build_obs_dict("F3", [None], 0),
        ]
        for obs in obsdic_list:
            fx = obs["type"]
            datapaths.append(pathlib.Path(f"DATA_PROTONBC_{fx}"))
        generate_empty(
            datapaths, destination, kin=KIN_DESC, match_type="KIN_PROTONBC"
        )

    dump_info_file(destination, "PROTONBC", obsdic_list, 1, M_PROTON)


def kinamatics_grids(is_xsec: bool, kin: dict) -> Tuple[dict, int]:
    """Generate the kinematic grids"""
    if is_xsec:
        n_xgrid = N_KINEMATC_GRID_XSEC["x"]
        n_q2grid = N_KINEMATC_GRID_XSEC["Q2"]
        n_ygrid = N_KINEMATC_GRID_XSEC["y"]
        y_grid = np.linspace(kin["y_min"], kin["y_max"], n_ygrid)
    else:
        n_xgrid = N_KINEMATC_GRID_FX["x"]
        n_q2grid = N_KINEMATC_GRID_FX["Q2"]
        n_ygrid = N_KINEMATC_GRID_FX["y"]
        y_grid = [0.0]

    x_grid = make_lambert_grid(n_xgrid, kin["x_min"])
    q2_grid = np.linspace(kin["q2_min"], kin["q2_max"], int(n_q2grid))
    n_points = int(n_q2grid * n_ygrid * n_xgrid)
    kin_grid = {"x": x_grid, "q2": q2_grid, "y": y_grid}
    return kin_grid, n_points


def dump_kinematics(
    destination: pathlib.Path, kin_grid: dict, match_type: str, is_xsec: bool
) -> None:
    """Dump the kinematics into CSV"""
    kinematics = {"x": [], "Q2": [], "y": []}
    for x, q2, y in itertools.product(
        kin_grid["x"], kin_grid["q2"], kin_grid["y"]
    ):
        kinematics["x"].append(x)
        kinematics["Q2"].append(q2)
        kinematics["y"].append(y)

    kinematics_pd = pd.DataFrame(kinematics)
    kinematics_pd.loc[-1] = ["mid", "mid", "mid"]
    kinematics_pd.index = kinematics_pd.index + 1
    kinematics_pd = kinematics_pd.sort_index()
    kinematics_folder = destination.joinpath("kinematics")
    kinematics_folder.mkdir(exist_ok=True)
    if is_xsec:
        write_to_csv(kinematics_folder, f"{match_type}_XSEC", kinematics_pd)
    else:
        write_to_csv(kinematics_folder, f"{match_type}_FX", kinematics_pd)


def dump_uncertainties(
    destination: pathlib.Path, name: str, n_points: int
) -> None:
    """Dump empty uncertainty files into CSV"""
    err_list = [{"stat": 0.0, "syst": 0.0}] * n_points
    errors_pd = construct_uncertainties(err_list)
    systypes_folder = destination.joinpath("uncertainties")
    systypes_folder.mkdir(exist_ok=True)
    write_to_csv(systypes_folder, f"UNC_{name}", errors_pd)


def main(
    grids: pathlib.Path,
    pdf: str,
    destination: pathlib.Path,
    kin: dict = KIN_DESC,
    match_type: str = "KIN_MATCHING",
) -> None:
    """Generate the Yadism data (kinematics & central values) as
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
        kin_grid, n_points = kinamatics_grids(is_xsec, kin)

        # get predictions
        full_pred = []
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            grid = pineappl.grid.Grid.read(gpath)
            prediction = pdf_error(grid, pdf, kin_grid["x"], reshape=False)
            full_pred.append(prediction[0])
        pred = np.average(full_pred, axis=0)

        # Select only predictions for Replicas_0 in data
        data_pd = pd.DataFrame({"data": pred[:, 0]})

        # Dump the kinematics into CSV
        dump_kinematics(destination, kin_grid, match_type, is_xsec)

        # Dump the central (replica) data into CSV
        central_val_folder = destination.joinpath("data")
        central_val_folder.mkdir(exist_ok=True)
        write_to_csv(central_val_folder, f"DATA_{new_name}", data_pd)

        # Dump the dummy uncertainties into CSV
        dump_uncertainties(destination, new_name, n_points)

        # Dump the predictions for the REST of the replicas as NPY
        pred_folder = destination.joinpath("matching")
        pred_folder.mkdir(exist_ok=True)
        mat_dest = (pred_folder / f"MATCH_{grid_name}").with_suffix(".npy")
        np.save(mat_dest, pred)

        msg = f"The matching/BC grid for {grid_name} are stored in "
        msg += f"'{destination.absolute().relative_to(pathlib.Path.cwd())}'"
        _logger.info(msg)


def generate_empty(
    datapaths: list[pathlib.Path],
    destination: pathlib.Path,
    kin: dict = KIN_DESC,
    match_type: str = "KIN_MATCHING",
) -> None:
    """Generate the empty matching datasets, with only kinematics table filled.

    Parameters
    ----------
    datapaths : list[pathlib.Path]
        list of datasets for which the matching tables are generated
    destination : pathlib.Path
        destination to store the files
    """
    destination.mkdir(parents=True, exist_ok=True)

    for dataset in datapaths:
        data_name = dataset.stem.strip("DATA_")
        if "MATCHING" in data_name:
            continue
        obs = data_name.split("_")[-1]
        new_name = f"{data_name}_MATCHING"

        is_xsec = "DXDY" in obs or "FW" in obs
        kin_grid, n_points = kinamatics_grids(is_xsec, kin)

        dump_kinematics(destination, kin_grid, match_type, is_xsec)

        # dump empty central values
        data_pd = pd.DataFrame({"data": np.zeros(n_points)})
        central_val_folder = destination.joinpath("data")
        central_val_folder.mkdir(exist_ok=True)
        write_to_csv(central_val_folder, f"DATA_{new_name}", data_pd)

        dump_uncertainties(destination, new_name, n_points)

        msg = f"The matching grid for {data_name} are stored in "
        msg += f"'{destination.absolute().relative_to(pathlib.Path.cwd())}'"
        _logger.info(msg)
