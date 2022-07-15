# -*- coding: utf-8 -*-
"""Utilities to write raw data filters."""
from pathlib import Path
from typing import Optional

import pandas as pd

ERR_DESC = {
    "stat": {
        "treatment": "ADD",
        "type": "UNCORR",
        "description": "Total statistical uncertainty",
    },
    "syst": {
        "treatment": "ADD",
        "type": "CORR",
        "description": "Total systematic uncertainty",
    },
}


def write_to_csv(path: Path, exp_name: str, table: pd.DataFrame):
    """Dump table to csv.

    Parameters
    ----------
    path: pathlib.Path
        path to the folder to dump into
    exp_name: str
        experiment name (i.e. the file stem)
    table: pd.DataFrame
        the table to dump

    """
    table.to_csv(f"{path}/{exp_name}.csv", encoding="utf-8")


def construct_uncertainties(full_obs_errors: list[float]) -> pd.DataFrame:
    """Load uncertainties from columns.

    Parameters
    ----------
    full_obs_errors: list[float]
        error columns to be properly loaded

    Returns
    -------
    pd.DataFrame
        actual table of uncertainties

    """
    header_struct = pd.MultiIndex.from_tuples(
        [(k, v["treatment"], v["type"]) for k, v in ERR_DESC.items()],
        names=["name", "treatment", "type"],
    )
    full_error_values = pd.DataFrame(full_obs_errors).values
    errors_pandas_table = pd.DataFrame(
        full_error_values,
        columns=header_struct,
        index=range(1, len(full_obs_errors) + 1),
    )
    return errors_pandas_table


def build_obs_dict(fx: str, tables: list[pd.DataFrame], pid: int) -> dict:
    """Add proper keys to arguments.

    Parameters
    ----------
    fx: str
        observable type
    tables: list
        data tables
    pid: int
        projectile PID

    Returns
    -------
    dict
        collection of arguments with proper keys provided

    """
    return {"type": fx, "tables": tables, "projectile": pid}


def dump_info_file(
    path: Path,
    exp_name: str,
    obs_list: list,
    target: Optional[float] = None,
    nucleon_mass: Optional[float] = 0.938,
):
    """Generate and dump info file.

    Parameters
    ----------
    path: pathlib.Path
        path to data folder, where to scope the info folder
    exp_name: str
        name of experiment (actual stem of the file)
    obs_list: list
        collection of observables
    target: None or float
        atomic number of the nuclear target

    """
    info_folder = path / "info"
    info_folder.mkdir(exist_ok=True)
    df = pd.DataFrame(obs_list)
    df["target"] = target
    df["m_nucleon"] = nucleon_mass
    df.to_csv(info_folder / f"{exp_name}.csv", encoding="utf-8")
