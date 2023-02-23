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

# Mass determined using Fe pdg values
M_NEUTRON = 939.565346 * 0.001
M_PROTON = 938.272013 * 0.001
A = 56  # A(Fe): Atomic Mass
Z = 26  # Z(Fe): Atomic Number
M_NUCLEON = 55.845 * 0.93149432 / (Z * M_PROTON + (A - Z) * M_NEUTRON)

# Experiment metadata
TARGET = A
EXP_NAME = "NUTEV"


def extract_sf(
    path: Path, exp_name: str, table_id_list: list, sfunc: str
) -> None:
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
    f2_exp_errors = []
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
        dep_var_f2dic = load_table["dependent_variables"][0]  # F_i
        sf_x_value = float(dep_var_f2dic["qualifiers"][2]["value"])
        # The numbers of bins should match the number of values
        # contained in the `independent_variables`. Now we can
        # loop over the different BINs
        for bin in range(len(indep_var_dic[0]["values"])):
            # ---- Extract only input kinematics ---- #
            q2_value = indep_var_dic[0]["values"][bin]["value"]
            kin_dict = {
                "x": {"mid": sf_x_value, "min": None, "max": None},
                "Q2": {"mid": q2_value, "min": None, "max": None},
                "y": {"mid": None, "min": None, "max": None},
            }
            kinematics.append(kin_dict)
            # ---- Extract central values for SF ---- #
            f2_value = dep_var_f2dic["values"][bin]["value"]
            f2_central.append(f2_value)
            # ---- Extract SYS & STAT uncertainties ---- #
            # Here we sum the systematic uncertainties over
            syst_dic = dep_var_f2dic["values"][bin]["errors"]
            syst = sum(
                syst_dic[i]["symerror"] for i in range(1, len(syst_dic) - 1)
            )
            error_dict_f2 = {
                "stat": dep_var_f2dic["values"][bin]["errors"][0]["symerror"],
                "syst": syst,
            }
            f2_exp_errors.append(error_dict_f2)

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

    # Convert the error dictionaries into Pandas tables
    f2_errors_pd = construct_uncertainties(f2_exp_errors)

    # Dump everything into files. In the following, F2 and xF3 lie on the central
    # values and errors share the same kinematic information and the difference.
    kinematics_folder = path.joinpath("kinematics")
    kinematics_folder.mkdir(exist_ok=True)
    write_to_csv(kinematics_folder, f"KIN_{exp_name}_{sfunc}", kinematics_pd)

    central_val_folder = path.joinpath("data")
    central_val_folder.mkdir(exist_ok=True)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_{sfunc}", f2pd)

    systypes_folder = path.joinpath("uncertainties")
    systypes_folder.mkdir(exist_ok=True)
    write_to_csv(systypes_folder, f"UNC_{exp_name}_{sfunc}", f2_errors_pd)


def extract_d2sigDxDy(
    path: Path, exp_name: str, table_id_list: list, obs: str
) -> None:
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
    dsig_nu_errors = []
    console.print(
        "\n• Extracting D2SIG/DX/DY from HEP tables:", style="bold blue"
    )
    # Loop over the tables that only contains the dsig/dx/dy
    for table_id in track(table_id_list, description="Progress tables"):
        table_path = path.joinpath(f"rawdata/{exp_name}/Table{table_id}.yaml")
        load_table = yaml.safe_load(table_path.read_text())
        # Extract the dictionary containing the high-level kinematic information
        indep_var_dic = load_table["independent_variables"]
        dependent_variables = load_table["dependent_variables"]
        nb_observables_classes = len(dependent_variables)
        for i in range(nb_observables_classes):
            dsig_dx_dy = dependent_variables[i]
            dsig_E_value = float(dsig_dx_dy["qualifiers"][0]["value"])
            dsig_x_value = float(dsig_dx_dy["qualifiers"][2]["value"])
            for bin in range(len(indep_var_dic[0]["values"])):
                # ---- Extract only input kinematics ---- #
                y = indep_var_dic[0]["values"][bin]["value"]
                # According to the paper, Q2=2M_p*x*y*E_nu
                q2_value = 2 * M_PROTON * dsig_x_value * y * dsig_E_value
                kin_dict = {
                    "x": {"mid": dsig_x_value, "min": None, "max": None},
                    "Q2": {"mid": q2_value, "min": None, "max": None},
                    "y": {"mid": y, "min": None, "max": None},
                }
                kinematics.append(kin_dict)
                # ---- Extract central values for SF ---- #
                dsig_nu_central.append(dsig_dx_dy["values"][bin]["value"])
                unc_type = dsig_dx_dy["values"][bin].get("errors", None)
                if unc_type is None:
                    stat_unc, syst_unc = 0.0, 0.0
                else:
                    stat_unc = unc_type[0]["symerror"]
                    syst_unc = unc_type[1]["symerror"]
                error_dict_1stentry = {"stat": stat_unc, "syst": syst_unc}
                dsig_nu_errors.append(error_dict_1stentry)

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
    dnuupd = pd.DataFrame(
        dsig_nu_central, index=range(1, nval_dnuu), columns=["data"]
    )
    dnuupd.index.name = "index"

    # Convert the error dictionaries into Pandas tables
    dsignuu_errors_pd = construct_uncertainties(dsig_nu_errors)

    # Dump everything into files. In the following, F2 and xF3 lie on the central
    # values and errors share the same kinematic information and the difference.
    kinematics_folder = path.joinpath("kinematics")
    kinematics_folder.mkdir(exist_ok=True)
    write_to_csv(kinematics_folder, f"KIN_{exp_name}_DXDY{obs}", kinematics_pd)

    central_val_folder = path.joinpath("data")
    central_val_folder.mkdir(exist_ok=True)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_DXDY{obs}", dnuupd)

    systypes_folder = path.joinpath("uncertainties")
    systypes_folder.mkdir(exist_ok=True)
    write_to_csv(
        systypes_folder, f"UNC_{exp_name}_DXDY{obs}", dsignuu_errors_pd
    )


def main(path_to_commondata: Path) -> None:
    """
    Parameters
    ----------
    path_to_commondata : Path
        path to the commondata folder
    """

    obs_list = []
    # List of tables containing measurements for F2
    table_f2 = [i for i in range(1, 13)]
    obs_list.append(build_obs_dict("F2", table_f2, 0))
    extract_sf(path_to_commondata, EXP_NAME, table_f2, "F2")

    # List of tables containing measurements for xF3
    table_f3 = [i for i in range(13, 25)]
    obs_list.append(build_obs_dict("F3", table_f3, 0))
    extract_sf(path_to_commondata, EXP_NAME, table_f3, "F3")

    # List of tables containing measurements for D2SIG/DX/DY for NUMU + FE
    table_dsig_dxdynuu = [i for i in range(26, 93)]
    obs_list.append(build_obs_dict("DXDYNUU", table_dsig_dxdynuu, 14))
    extract_d2sigDxDy(path_to_commondata, EXP_NAME, table_dsig_dxdynuu, "NUU")

    # List of tables containing measurements for D2SIG/DX/DY for NUBMU + FE
    table_dsig_dxdynub = [i for i in range(93, 160)]
    obs_list.append(build_obs_dict("DXDYNUB", table_dsig_dxdynub, -14))
    extract_d2sigDxDy(path_to_commondata, EXP_NAME, table_dsig_dxdynub, "NUB")

    # dump info file
    dump_info_file(path_to_commondata, EXP_NAME, obs_list, TARGET, M_PROTON)


if __name__ == "__main__":
    relative_path = Path().absolute().parents[3].joinpath("commondata")
    main(relative_path)
