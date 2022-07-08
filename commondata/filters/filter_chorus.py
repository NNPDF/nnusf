#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Mass determined using scikithep/particle for Pb208 in GeV
M_NEUTRON = 939.565346 * 0.001
M_PROTON = 938.272013 * 0.001
A = 208  # A(Pb): Atomic Mass
Z = 82  # Z(Pb): Atomic Number
M_NUCLEON = 193.729 / (Z * M_PROTON + (A - Z) * M_NEUTRON)

# Experiment Metadata
TARGET = A
EXP_NAME = "CHORUS"


def extract_f2f3(path: Path, exp_name: str, table_id_list: list) -> None:
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
        f2_x_value = float(dep_var_f2dic["qualifiers"][4]["value"])
        f3_x_value = float(dep_var_f3dic["qualifiers"][4]["value"])
        assert f2_x_value == f3_x_value
        # The numbers of bins should match the number of values
        # contained in the `independent_variables`. Now we can
        # loop over the different BINs
        for bin in range(len(indep_var_dic[0]["values"])):
            # ---- Extract only input kinematics ---- #
            q2_value = indep_var_dic[0]["values"][bin]["value"]
            kin_dict = {
                "x": {"mid": f2_x_value, "min": None, "max": None},
                "Q2": {"mid": q2_value, "min": None, "max": None},
                "y": {"mid": None, "min": None, "max": None},
            }
            kinematics.append(kin_dict)
            # ---- Extract central values for SF ---- #
            f2_value = dep_var_f2dic["values"][bin]["value"]
            f2_central.append(f2_value)
            f3_value = dep_var_f3dic["values"][bin]["value"]
            f3_central.append(f3_value)
            # ---- Extract SYS & STAT uncertainties ---- #
            error_dict_f2 = {
                "stat": dep_var_f2dic["values"][bin]["errors"][0]["symerror"],
                "syst": dep_var_f2dic["values"][bin]["errors"][1]["symerror"],
            }
            f2_exp_errors.append(error_dict_f2)
            error_dict_f3 = {
                "stat": dep_var_f3dic["values"][bin]["errors"][0]["symerror"],
                "syst": dep_var_f3dic["values"][bin]["errors"][1]["symerror"],
            }
            f3_exp_errors.append(error_dict_f3)

    # Convert the kinematics dictionaries into Pandas tables
    full_kin = {
        i + 1: pd.DataFrame(d).stack() for i, d in enumerate(kinematics)
    }
    kinematics_pd = (
        pd.concat(
            full_kin,
            axis=1,
        )
        .swaplevel(0, 1)
        .T
    )

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


def extract_d2sigDxDy(path: Path, exp_name: str, table_id_list: list) -> None:
    """Extract the double differential cross section.

    Parameters
    ----------
    path : Path
        Path to the commondata folder
    exp_name : str
        name of the experiment
    table_id_list : list
        list of table that corresponds to DSIG/DX/DY
    """

    kinematics = []
    dsig_nu_central = []
    dsig_nub_centrl = []
    dsig_nu_errors = []
    dsig_nub_error = []
    console.print(
        "\n• Extracting D2SIG/DX/DY from HEP tables:", style="bold blue"
    )
    # Loop over the tables that only contains the dsig/dx/dy
    for table_id in track(table_id_list, description="Progress tables"):
        table_path = path.joinpath(f"rawdata/{exp_name}/Table{table_id}.yaml")
        load_table = yaml.safe_load(table_path.read_text())
        # Extract the dictionary containing the high-level
        # kinematic information
        indep_var_dic = load_table["independent_variables"]
        dep_var_dsig_nu = load_table["dependent_variables"][0]  # NU
        dep_vr_dsig_nub = load_table["dependent_variables"][1]  # NUB
        # The x values should be the same for dsignu & dsignub
        dsignuu_x_value = float(dep_var_dsig_nu["qualifiers"][3]["value"])
        dsignub_x_value = float(dep_vr_dsig_nub["qualifiers"][3]["value"])
        assert dsignuu_x_value == dsignub_x_value
        # The x values should be the same for dsignu & dsignub
        dsignuu_Enu_value = float(dep_var_dsig_nu["qualifiers"][0]["value"])
        dsignub_Enu_value = float(dep_vr_dsig_nub["qualifiers"][0]["value"])
        assert dsignuu_Enu_value == dsignub_Enu_value
        # Extract the value of the COM energy (should be the same for DXDY)
        dsignuu_sqrts = float(dep_var_dsig_nu["qualifiers"][2]["value"])
        dsignub_sqrts = float(dep_vr_dsig_nub["qualifiers"][2]["value"])
        assert dsignuu_sqrts == dsignub_sqrts
        # The numbers of bins should match the number of values
        # contained in the `independent_variables`. Now we can
        # loop over the different BINs
        for bin in range(len(indep_var_dic[0]["values"])):
            # ---- Extract only input kinematics ---- #
            y = indep_var_dic[0]["values"][bin]["value"]
            # According to the paper Q2 = 2*x*y*M_N*E_nu [GeV^2]
            q2_value = 2 * dsignuu_x_value * y * M_NUCLEON * dsignuu_Enu_value
            kin_dict = {
                "x": {"mid": dsignuu_x_value, "min": None, "max": None},
                "Q2": {"mid": q2_value, "min": None, "max": None},
                "y": {"mid": y, "min": None, "max": None},
            }
            kinematics.append(kin_dict)
            # ---- Extract central values for SF ---- #
            # Note the additional factor to convert the xs to cm^2/Gev
            dsig_nuu_value = dep_var_dsig_nu["values"][bin]["value"]
            dsig_nu_central.append(float(dsig_nuu_value) / 10)
            dsig_nub_value = dep_vr_dsig_nub["values"][bin]["value"]
            dsig_nub_centrl.append(float(dsig_nub_value) / 10)
            # ---- Extract SYS & STAT uncertainties ---- #
            error_dict_1stentry = {
                "stat": float(
                    dep_var_dsig_nu["values"][bin]["errors"][0]["symerror"]
                )
                / 10,
                "syst": float(
                    dep_var_dsig_nu["values"][bin]["errors"][1]["symerror"]
                )
                / 10,
            }
            dsig_nu_errors.append(error_dict_1stentry)
            error_dict_2ndentry = {
                "stat": float(
                    dep_vr_dsig_nub["values"][bin]["errors"][0]["symerror"]
                )
                / 10,
                "syst": float(
                    dep_vr_dsig_nub["values"][bin]["errors"][1]["symerror"]
                )
                / 10,
            }
            dsig_nub_error.append(error_dict_2ndentry)

    # Convert the kinematics dictionaries into Pandas tables
    full_kin = {
        i + 1: pd.DataFrame(d).stack() for i, d in enumerate(kinematics)
    }
    kinematics_pd = (
        pd.concat(
            full_kin,
            axis=1,
        )
        .swaplevel(0, 1)
        .T
    )

    # Convert the central data values dict into Pandas tables
    nval_dnuu = len(dsig_nu_central) + 1
    nval_dnub = len(dsig_nub_centrl) + 1
    dnuupd = pd.DataFrame(
        dsig_nu_central, index=range(1, nval_dnuu), columns=["data"]
    )
    dnuupd.index.name = "index"
    dnubpd = pd.DataFrame(
        dsig_nub_centrl, index=range(1, nval_dnub), columns=["data"]
    )
    dnubpd.index.name = "index"

    # Convert the error dictionaries into Pandas tables
    dsignuu_errors_pd = construct_uncertainties(dsig_nu_errors)
    dsignub_errors_pd = construct_uncertainties(dsig_nub_error)

    # Dump everything into files. In the following, F2 and xF3 lie on the central
    # values and errors share the same kinematic information and the difference.
    kinematics_folder = path.joinpath("kinematics")
    kinematics_folder.mkdir(exist_ok=True)
    write_to_csv(kinematics_folder, f"KIN_{exp_name}_DXDY", kinematics_pd)

    central_val_folder = path.joinpath("data")
    central_val_folder.mkdir(exist_ok=True)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_DXDYNUU", dnuupd)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_DXDYNUB", dnubpd)

    systypes_folder = path.joinpath("uncertainties")
    systypes_folder.mkdir(exist_ok=True)
    write_to_csv(systypes_folder, f"UNC_{exp_name}_DXDYNUU", dsignuu_errors_pd)
    write_to_csv(systypes_folder, f"UNC_{exp_name}_DXDYNUB", dsignub_errors_pd)


def main(path_to_commondata: Path) -> None:
    """
    Parameters
    ----------
    path_to_commondata : Path
        path to the commondata folder
    """

    obs_list = []
    # List of tables containing measurements for F2 and xF3
    table_f2_xf3 = [i for i in range(1, 12)]
    obs_list.extend(
        [
            build_obs_dict("F2", table_f2_xf3, 0),
            build_obs_dict("F3", table_f2_xf3, 0),
        ]
    )
    extract_f2f3(path_to_commondata, EXP_NAME, table_f2_xf3)

    # List of tables containing measurements for D2SIG/DX/DY
    table_dsig_dxdy = [i for i in range(23, 122)]
    obs_list.extend(
        [
            build_obs_dict("DXDYNUU", table_dsig_dxdy, 14),
            build_obs_dict("DXDYNUB", table_dsig_dxdy, -14),
        ]
    )
    extract_d2sigDxDy(path_to_commondata, EXP_NAME, table_dsig_dxdy)

    # dump info file
    dump_info_file(path_to_commondata, EXP_NAME, obs_list, TARGET, M_NUCLEON)


if __name__ == "__main__":
    relative_path = Path().absolute().parents[3].joinpath("commondata")
    main(relative_path)
