#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter for BEBCWA59 experiment."""

from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console
from rich.progress import track

from nnusf.data.utils import (
    build_obs_dict,
    construct_uncertainties,
    dump_info_file,
    write_to_csv,
)

console = Console()

# Mass determined using Ne pdg values
M_NEUTRON = 939.565346 * 0.001
M_PROTON = 938.272013 * 0.001
A = 20  # A(Ne): Atomic Mass
Z = 10  # Z(Ne): Atomic Number
M_NUCLEON = 20.1797 * 0.93149432 / (Z * M_PROTON + (A - Z) * M_NEUTRON)


# Experiment metadata
TARGET = A
EXP_NAME = "BEBCWA59"


def extract_f2f3(path: Path, exp_name: str, table_id_list: list):
    """Extract F2 and xF3 structure functions.

    Parameters
    ----------
    path : Path
        Path to the commondata folder
    exp_name : str
        name of the experiment
    table_id_list : list
        list of table that corresponds to F2 & xF3

    """
    kinematics = []
    f2_central = []
    f3_central = []
    f2_exp_errors = []
    f3_exp_errors = []
    console.print(
        "\n• Extracting F2 and xF3 from HEP tables:", style="bold blue"
    )
    # Loop over the tables that only contains the F2/xF3
    for table_id in track(table_id_list, description="Progress tables"):
        table_path = path.joinpath(f"rawdata/{exp_name}/Table{table_id}.yaml")
        load_table = yaml.safe_load(table_path.read_text())
        # Extract the dictionary containing the high-level
        # kinematic information
        indep_var_dic = load_table["independent_variables"]
        dep_var_f2dic = load_table["dependent_variables"][0]  # F2
        dep_var_f3dic = load_table["dependent_variables"][1]  # xF3
        # The x values should be the same for F2 & xF3
        f2_q2_value = float(dep_var_f2dic["qualifiers"][0]["value"])
        f3_q2_value = float(dep_var_f3dic["qualifiers"][0]["value"])
        assert f2_q2_value == f3_q2_value
        # The numbers of bins should match the number of values
        # contained in the `independent_variables`. Now we can
        # loop over the different BINs
        for bin in range(len(indep_var_dic[0]["values"])):
            # ---- Extract only input kinematics ---- #
            x_value = indep_var_dic[0]["values"][bin]["value"]
            kin_dict = {
                "x": {"mid": x_value, "min": None, "max": None},
                "Q2": {"mid": f2_q2_value, "min": None, "max": None},
                "y": {"mid": None, "min": None, "max": None},
            }
            kinematics.append(kin_dict)
            # ---- Extract central values for SF ---- #
            f2_value = dep_var_f2dic["values"][bin]["value"]
            f2_central.append(f2_value)
            f3_value = dep_var_f3dic["values"][bin]["value"]
            f3_central.append(f3_value)
            # ---- Extract SYS & STAT uncertainties ---- #
            uncertainties_sfs = [
                dep_var_f2dic["values"][bin].get("errors", None),
                dep_var_f3dic["values"][bin].get("errors", None),
            ]
            uncertainty_dic, uncertainty_names = {}, ["f2_unc", "f3_unc"]
            for idx, unc_type in enumerate(uncertainties_sfs):
                if unc_type is None:
                    stat_unc, syst_unc = 0.0, 0.0
                else:
                    stat_unc = unc_type[0].get("symerror", 0.0)
                    syst_unc = unc_type[1].get("symerror", 0.0)
                uncertainty_dic[uncertainty_names[idx]] = [stat_unc, syst_unc]
            error_dict_f2 = {
                "stat": uncertainty_dic["f2_unc"][0],
                "syst": uncertainty_dic["f2_unc"][1],
            }
            f2_exp_errors.append(error_dict_f2)
            error_dict_f3 = {
                "stat": uncertainty_dic["f3_unc"][0],
                "syst": uncertainty_dic["f3_unc"][1],
            }
            f3_exp_errors.append(error_dict_f3)

    # Convert the kinematics dictionaries into Pandas tables
    full_kin = {
        i + 1: pd.DataFrame(d).stack() for i, d in enumerate(kinematics)
    }
    kinematics_pd = pd.concat(full_kin, axis=1).swaplevel(0, 1).T

    # Convert the central data values dict into Pandas tables
    f2pd = pd.DataFrame(
        f2_central, index=range(1, len(f2_central) + 1), columns=["data"]
    )
    f2pd.index.name = "index"
    f3pd = pd.DataFrame(
        f3_central, index=range(1, len(f3_central) + 1), columns=["data"]
    )
    f3pd.index.name = "index"

    # Convert the error dictionaries into Pandas tables
    f2_errors_pd = construct_uncertainties(f2_exp_errors)
    f3_errors_pd = construct_uncertainties(f3_exp_errors)

    # Dump everything into files. In the following, F2 and xF3 lie on the central
    # values and errors share the same kinematic information and the difference.
    kinematics_folder = path.joinpath("kinematics")
    kinematics_folder.mkdir(exist_ok=True)
    write_to_csv(kinematics_folder, f"KIN_{exp_name}_F2F3", kinematics_pd)

    central_val_folder = path.joinpath("data")
    central_val_folder.mkdir(exist_ok=True)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_F2", f2pd)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_F3", f3pd)

    systypes_folder = path.joinpath("uncertainties")
    systypes_folder.mkdir(exist_ok=True)
    write_to_csv(systypes_folder, f"UNC_{exp_name}_F2", f2_errors_pd)
    write_to_csv(systypes_folder, f"UNC_{exp_name}_F3", f3_errors_pd)


def main(path_to_commondata: Path):
    """Run filter.

    Parameters
    ----------
    path_to_commondata : Path
        path to the commondata folder

    """
    # List of tables containing measurements for F2 and xF3
    table_f2_xf3 = [i for i in range(1, 10)]
    obs_list = [
        build_obs_dict("F2", table_f2_xf3, 0),
        build_obs_dict("F3", table_f2_xf3, 0),
    ]
    extract_f2f3(path_to_commondata, EXP_NAME, table_f2_xf3)

    # dump info file
    dump_info_file(path_to_commondata, EXP_NAME, obs_list, TARGET, M_NUCLEON)


if __name__ == "__main__":
    relative_path = Path().absolute().parents[3] / "commondata"
    main(relative_path)
