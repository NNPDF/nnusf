#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
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
EXP_NAME = "CDHSW"


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
        f2_x_value = float(dep_var_f2dic["qualifiers"][2]["value"])
        f3_x_value = float(dep_var_f3dic["qualifiers"][2]["value"])
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
            uncertainties_sfs = [
                dep_var_f2dic["values"][bin].get("errors", None),
                dep_var_f3dic["values"][bin].get("errors", None),
            ]
            uncertainty_dic, uncertainty_names = {}, ["f2_unc", "f3_unc"]
            for idx, unc_type in enumerate(uncertainties_sfs):
                if unc_type is None:
                    stat_unc, syst_unc = 0.0, 0.0
                else:
                    syst_unc = 0.0
                    for unc in unc_type:
                        # stat
                        if unc["label"] == "stat":
                            stat_unc = unc["symerror"]
                        if unc["label"] == "sys":
                            # asymmetric sys
                            if "asymerror" in unc:
                                syst_unc += (
                                    np.abs(
                                        [
                                            unc["asymerror"]["plus"],
                                            unc["asymerror"]["minus"],
                                        ]
                                    ).max()
                                    ** 2
                                )
                            # symmetric sys
                            else:
                                syst_unc += unc["symerror"] ** 2
                    syst_unc = np.sqrt(syst_unc)
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


def extract_d2sigDxDy(
    path: Path, exp_name: str, table_id_list: list, obs: str
) -> None:
    """Extract the double differential cross sections.

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
            # Extract all depdendent variables in order to compute Q2
            dsig_E_value = float(dsig_dx_dy["qualifiers"][0]["value"])
            # dsig_sqrts = float(dsig_dx_dy["qualifiers"][2]["value"])
            dsig_Y_bins = dsig_dx_dy["qualifiers"][3]["value"].split()
            y_value_min = float(dsig_Y_bins[0])
            y_value_max = float(dsig_Y_bins[-1])
            y_value_mid = (y_value_max + y_value_min) / 2
            for bin in range(len(indep_var_dic[0]["values"])):
                # ---- Extract only input kinematics ---- #
                x_valmax = indep_var_dic[0]["values"][bin]["high"]
                x_valmin = indep_var_dic[0]["values"][bin]["low"]
                x_valmid = (x_valmax + x_valmin) / 2
                # Computing Q2 according to the paper, Q2=2M_N*x*y*E_nu
                q2_min = 2 * M_NUCLEON * x_valmin * y_value_min * dsig_E_value
                q2_mid = 2 * M_NUCLEON * x_valmid * y_value_mid * dsig_E_value
                q2_max = 2 * M_NUCLEON * x_valmax * y_value_max * dsig_E_value
                kin_dict = {
                    "x": {"mid": x_valmid, "min": x_valmin, "max": x_valmax},
                    "Q2": {"mid": q2_mid, "min": q2_min, "max": q2_max},
                    "y": {
                        "mid": y_value_mid,
                        "min": y_value_min,
                        "max": y_value_max,
                    },
                }
                kinematics.append(kin_dict)
                # ---- Extract central values and uncertainties ---- #
                dsig_nu_central.append(dsig_dx_dy["values"][bin]["value"])
                unc_type = dsig_dx_dy["values"][bin].get("errors", None)
                if unc_type is None:
                    stat_unc, syst_unc = 0.0, 0.0
                else:
                    stat_unc = unc_type[0].get("symerror", 0.0)
                    syst_unc = unc_type[1].get("symerror", 0.0)
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


def extract_fw(path: Path, exp_name: str, table_id_list: list) -> None:
    """Extract the F_W observable.

    Parameters
    ----------
    path : Path
        Path to the commondata folder
    exp_name : str
        name of the experiment
    table_id_list : list
        list of table that corresponds to F_W
    """

    kinematics = []
    fw_central = []
    fw_exp_errors = []
    console.print("\n• Extracting FW from HEP tables:", style="bold blue")
    # Loop over the tables that only contains the FW
    for table_id in track(table_id_list, description="Progress tables"):
        table_path = path.joinpath(f"rawdata/{exp_name}/Table{table_id}.yaml")
        load_table = yaml.safe_load(table_path.read_text())
        # Extract the dictionary containing the high-level
        # kinematic information
        indep_var_dic = load_table["independent_variables"]
        dep_var_Adic = load_table["dependent_variables"][0]  # A
        dep_var_fwdic = load_table["dependent_variables"][1]  # FW
        # The x values should be the same for FW
        fw_x_value = float(dep_var_fwdic["qualifiers"][2]["value"])
        # The numbers of bins should match the number of values
        # contained in the `independent_variables`. Now we can
        # loop over the different BINs
        for bin in range(len(indep_var_dic[0]["values"])):
            # ---- Extract only input kinematics ---- #
            q2_value = indep_var_dic[0]["values"][bin]["value"]
            A = dep_var_Adic["values"][bin]["value"]
            b = fw_x_value**2 * M_NUCLEON**2 / q2_value
            # calculate y
            sqrt = np.sqrt(4 * A**2 * b - A**2 + 2 * A)
            den = 2 * A * b - A + 1
            y1 = -(A + sqrt) / den
            y2 = -(A - sqrt) / den
            if y1 >= 0 and y2 >= 0:
                raise ValueError("y values are both positive")
            kin_dict = {
                "x": {"mid": fw_x_value, "min": None, "max": None},
                "Q2": {"mid": q2_value, "min": None, "max": None},
                "y": {"mid": max(y1, y2), "min": None, "max": None},
            }
            kinematics.append(kin_dict)
            # ---- Extract central values for SF ---- #
            fw_value = dep_var_fwdic["values"][bin]["value"]
            fw_central.append(fw_value)
            # ---- Extract SYS & STAT uncertainties ---- #
            uncertainties_sfs = [
                dep_var_fwdic["values"][bin].get("errors", None),
            ]
            uncertainty_dic, uncertainty_name = {}, "fw_unc"
            for unc_type in uncertainties_sfs:
                if unc_type is None:
                    stat_unc, syst_unc = 0.0, 0.0
                else:
                    syst_unc = 0.0
                    for unc in unc_type:
                        # stat
                        if unc["label"] == "stat":
                            stat_unc = unc["symerror"]
                        if unc["label"] == "sys":
                            # asymmetric sys
                            if "asymerror" in unc:
                                syst_unc += (
                                    np.abs(
                                        [
                                            unc["asymerror"]["plus"],
                                            unc["asymerror"]["minus"],
                                        ]
                                    ).max()
                                    ** 2
                                )
                            # symmetric sys
                            else:
                                syst_unc += unc["symerror"] ** 2
                    syst_unc = np.sqrt(syst_unc)
                uncertainty_dic[uncertainty_name] = [stat_unc, syst_unc]
            error_dict_fw = {
                "stat": uncertainty_dic["fw_unc"][0],
                "syst": uncertainty_dic["fw_unc"][1],
            }
            fw_exp_errors.append(error_dict_fw)

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
    fwpd = pd.DataFrame(
        fw_central, index=range(1, len(fw_central) + 1), columns=["data"]
    )
    fwpd.index.name = "index"

    # Convert the error dictionaries into Pandas tables
    fw_errors_pd = construct_uncertainties(fw_exp_errors)

    # Dump everything into files. In the following, FW lie on the central
    # values and errors share the same kinematic information and the difference.
    kinematics_folder = path.joinpath("kinematics")
    kinematics_folder.mkdir(exist_ok=True)
    write_to_csv(kinematics_folder, f"KIN_{exp_name}_FW", kinematics_pd)

    central_val_folder = path.joinpath("data")
    central_val_folder.mkdir(exist_ok=True)
    write_to_csv(central_val_folder, f"DATA_{exp_name}_FW", fwpd)

    systypes_folder = path.joinpath("uncertainties")
    systypes_folder.mkdir(exist_ok=True)
    write_to_csv(systypes_folder, f"UNC_{exp_name}_FW", fw_errors_pd)


def main(path_to_commondata: Path) -> None:
    """
    Parameters
    ----------
    path_to_commondata : Path
        path to the commondata folder
    """

    obs_list = []

    # List of tables containing measurements for F2 and xF3
    table_f2_xf3 = [i for i in range(19, 30)]
    obs_list.extend(
        [
            build_obs_dict("F2", table_f2_xf3, 14),
            build_obs_dict("F3", table_f2_xf3, 14),
        ]
    )
    extract_f2f3(path_to_commondata, EXP_NAME, table_f2_xf3)

    # List of tables containing measurements for D2SIG/DX/DY for NUMU + FE
    table_dsig_dxdynuu = [i for i in range(1, 10)]
    obs_list.append(build_obs_dict("DXDYNUU", table_dsig_dxdynuu, 14))
    extract_d2sigDxDy(path_to_commondata, EXP_NAME, table_dsig_dxdynuu, "NUU")

    # List of tables containing measurements for D2SIG/DX/DY for NUMUB + FE
    table_dsig_dxdynub = [i for i in range(10, 18)]
    obs_list.append(build_obs_dict("DXDYNUB", table_dsig_dxdynub, -14))
    extract_d2sigDxDy(path_to_commondata, EXP_NAME, table_dsig_dxdynub, "NUB")

    # List of tables containing measurements for FW
    table_fw = [i for i in range(30, 40)]
    obs_list.append(build_obs_dict("FW", table_fw, -14))
    extract_fw(path_to_commondata, EXP_NAME, table_fw)
    # dump info file
    dump_info_file(path_to_commondata, EXP_NAME, obs_list, TARGET, M_NUCLEON)


if __name__ == "__main__":
    relative_path = Path().absolute().parents[3].joinpath("commondata")
    main(relative_path)
