# -*- coding: utf-8 -*-
"""Utilities to write raw data filters."""
import pathlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
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

MAP_EXP_YADISM = {
    "NUTEV": "XSNUTEVNU",
    "CHORUS": "XSCHORUSCC",
    "CDHSW": "XSCHORUSCC",
    "PROTONBC": "XSCHORUSCC",
}

MAP_OBS_PID = {"F2": 0, "F3": 0, "DXDYNUU": 14, "DXDYNUB": -14}


class ObsTypeError(Exception):
    """Raised when observable is not recognized."""


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


def build_obs_dict(
    fx: str, tables: list[Union[pd.DataFrame, None]], pid: int
) -> dict:
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


def parse_input_kinematics(
    mainpath: pathlib.Path,
    name: str,
    info_name: str,
    exp_name: str,
    obs: str,
):
    """Read the pandas tables that contain the input kinematic values.

    Parameters
    ----------
    mainpath: pathlib.Path
        path to the kinematics
    name: str
        name of the dataset
    info_name: str
        name
    exp_name: str
        name of the experiment
    obs: str
        Type of the observable

    Returns
    -------
    pd.DataFrame
        tables containing the kinematic info
    """

    kin_file = mainpath.joinpath(f"kinematics/KIN_{name}.csv")

    if kin_file.exists():
        kin_df = pd.read_csv(kin_file).iloc[1:, 1:4].reset_index(drop=True)
    elif "_MATCHING" in name:
        file_path = f"{mainpath}/kinematics"
        if "FW" in name or "DXDY" in name:
            file = f"{file_path}/KIN_{info_name}_MATCHING_DXDY.csv"
        else:
            file = f"{file_path}/KIN_{info_name}_MATCHING_F2F3.csv"
        kin_df = pd.read_csv(file).iloc[1:, 1:4].reset_index(drop=True)
    elif obs in ["F2", "F3"]:
        file = f"{mainpath}/kinematics/KIN_{exp_name}_F2F3.csv"
        kin_df = pd.read_csv(file).iloc[1:, 1:4].reset_index(drop=True)
    elif obs in ["DXDYNUU", "DXDYNUB"]:
        file = f"{mainpath}/kinematics/KIN_{exp_name}_DXDY.csv"
        kin_df = pd.read_csv(file).iloc[1:, 1:4].reset_index(drop=True)
    else:
        raise ObsTypeError(f"{obs} is not recognised as an Observable.")

    return kin_df


def parse_central_values(mainpath: pathlib.Path, name: str):
    """Extract the central values of either the experimental datasets
    or the matching pseudodata.

    Parameters
    ----------
    mainpath: pathlib.Path
        path to the (pseudo)dataset
    name: str
        name of the dataset

    Returns
    -------
    pd.DataFrame
        table containing the central values
    """

    dat_name = f"{mainpath}/data/DATA_{name}.csv"
    data_df = pd.read_csv(dat_name, header=0, na_values=["-", " "])
    data_df = data_df.iloc[:, 1:].reset_index(drop=True)

    return data_df


def parse_uncertainties(mainpath: pathlib.Path, name: str):
    """Extract the central values of either the experimental datasets
    or the matching pseudodata.

    Parameters
    ----------
    mainpath: pathlib.Path
        path to the (pseudo)dataset
    name: str
        name of the dataset

    Returns
    -------
    pd.DataFrame
        table containing the central values
    """

    unc_name = f"{mainpath}/uncertainties/UNC_{name}.csv"
    unc_df = pd.read_csv(unc_name, na_values=["-", " "])
    unc_df = unc_df.iloc[2:, 1:].reset_index(drop=True)

    return unc_df


def add_w2_table(table: pd.DataFrame, m_nucleus: int):
    """Compute the value of the W2 given the input kinematics.

    Parameters
    ----------
    table: pd.DataFrame
        table containing all the data specs
    m_nucleus: int
        value of the nucleon/nucleus mass

    Returns:
    --------
    pd.DataFrame
        table with W2 column appended
    """

    q2 = table["Q2"].astype(float, errors="raise")  # Object -> float
    xx = table["x"].astype(float, errors="raise")  # Object -> float
    table["W2"] = q2 * (1 - xx) / xx + m_nucleus

    return table


def combine_tables(
    kin_df: pd.DataFrame, data_df: pd.DataFrame, unc_df: pd.DataFrame, name: str
):
    """Combined the kinematic, central values, and uncertainty tables
    into one making sure that datasets with total zero uncertainties
    are dropped.

    Parameters
    ----------
    kin_df: pd.DataFrame
        table containing the kinematic values
    data_df: pd.DataFrame
        table containing the central values
    unc_df:
        table containing the uncertainty values
    name: str
        name of the dataset

    Returns:
    --------
    pd.DataFrame
        combined table with index reset
    """
    # Concatenate enverything into one single big table
    new_df = pd.concat([kin_df, data_df, unc_df], axis=1)
    new_df = new_df.dropna().astype(float)

    # drop data with 0 total uncertainty:
    if "_MATCHING" not in name:
        new_df = new_df[new_df["stat"] + new_df["syst"] != 0.0]

    # Make sure that the indices are restored for the cuts
    new_df.reset_index(drop=True, inplace=True)

    return new_df


def apply_cuts(new_df: pd.DataFrame, name: str, kincuts: dict):
    """Given a dictionary containing information on the cuts, apply
    them to the entire table. There are at most three main cuts that
    can be applied:

    Parameters
    ----------
    new_df: pd.DataFrame
        combined table contaiing all the dataset specifications
    name: str
        name of the dataset
    kincuts: dict
        dictionary containing all the specifities of the cuts

    Returns
    -------
    pd.DataFrame
        table with cuts applied
    """

    # Extract values of kinematic cuts if any
    w2min = kincuts.get("w2min", None)
    q2max = kincuts.get("q2max", None)

    # Perform cuts along the W2 direction
    new_df = new_df[new_df["W2"] >= w2min] if w2min else new_df

    # For real datasets we also need to impose a maximum Q2 cut
    if "_MATCHING" not in name:
        new_df = new_df[new_df["Q2"] <= q2max] if q2max else new_df

    return new_df


def append_target_info(
    new_df: pd.DataFrame, info_df: pd.DataFrame, exp_name: str, obs: str
):
    """Append the information regarding the nuclon/nucleus to the table.

    Parameters
    ----------
    new_df: pd.DataFrame
        combined table containing the dataset specifications
    info_df: pd.DataFrame
        table containing the info regarding the target
    exp_name: str
        name of the experiment
    obs: str
        type of the observable

    Returns
    -------
    pd.DataFrame
        table with all the target info included
    """

    # Extract the number of data points after the cuts for this dataset
    number_datapoints = new_df.shape[0]

    # Extract the information on the cross section (FW is a special case)
    data_spec = "FW" if obs == "FW" else MAP_EXP_YADISM.get(exp_name, None)

    # Append all the info columns to the `kin_df` table
    new_df["A"] = np.full(number_datapoints, info_df["target"][0])
    new_df["xsec"] = np.full(number_datapoints, data_spec)
    new_df["Obs"] = np.full(number_datapoints, obs)
    new_df["projectile"] = np.full(
        number_datapoints,
        info_df.loc[info_df["type"] == obs, "projectile"],
    )
    new_df["m_nucleon"] = np.full(number_datapoints, info_df["m_nucleon"][0])

    return new_df
