#!/usr/bin/env python3

import yaml
import pandas as pd

from pathlib import Path
from rich.console import Console
from rich.progress import track

ERR_DESC = {
    'stat': {
        'treatment': "ADD",
        'type': "UNCORR",
        'description': "Total statistical uncertainty"
    },
    'syst': {
        'treatment': "ADD",
        'type': "CORR",
        'description': "Total systematic uncertainty"
    }
}

console = Console()
M_NUCLEON = 0.938 # GeV


def write_to_csv(path: Path, exp_name:str, file: pd.DataFrame) -> None:
    file.to_csv(f"{path}/{exp_name}.csv", encoding="utf-8")


def construct_uncertainties(full_obs_errors: list) -> pd.DataFrame:
    header_struct = pd.MultiIndex.from_tuples(
        [(k, v["treatment"], v["type"]) for k,v in ERR_DESC.items()],
        names=["name", "treatment", "type"]
    )
    full_error_values = pd.DataFrame(full_obs_errors).values
    errors_pandas_table = pd.DataFrame(
        full_error_values,
        columns=header_struct,
        index=range(1, len(full_obs_errors) + 1)
    )
    errors_pandas_table.index.name = "index"
    return errors_pandas_table


def extract_f2f3(path: Path, exp_name: str, table_id_list: list) -> None:
    """
    Parameters:
    -----------
    path: Path
    """
    kinematics = []
    f2_central = []
    f3_central = []
    f2_exp_errors = []
    f3_exp_errors = []
    console.print("\n• Extracting F2 and xF3 from HEP tables:", style="bold blue")
    # Loop over the tables that only contains the F2/xF3
    for table_id in track(table_id_list, description="Progress tables"):
        table_path = path.joinpath(f"rawdata/{exp_name}/Table{table_id}.yaml")
        load_table = yaml.safe_load(table_path.read_text())
        # Extract the dictionary containing the high-level
        # kinematic information
        indep_var_dic = load_table["independent_variables"]
        dep_var_f2dic = load_table["dependent_variables"][0] # F2
        dep_var_f3dic = load_table["dependent_variables"][1] # xF3
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
                "x": {"min": None, "mid": f2_x_value, "max": None},
                "Q2": {"min": None, "mid": q2_value, "max": None},
                "y": {"min": None, "mid": None, "max": None}
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
                    stat_unc, syst_unc = None, None
                else:
                    stat_unc = unc_type[0].get("symerror", None)
                    syst_unc = unc_type[1].get("symerror", None)
                uncertainty_dic[uncertainty_names[idx]] = [stat_unc, syst_unc]
            error_dict_f2 = {
                "stat": uncertainty_dic["f2_unc"][0],
                "syst": uncertainty_dic["f2_unc"][1]
            }
            f2_exp_errors.append(error_dict_f2)
            error_dict_f3 = {
                "stat": uncertainty_dic["f3_unc"][0],
                "syst": uncertainty_dic["f3_unc"][1]
            }
            f3_exp_errors.append(error_dict_f3)

    # Convert the kinematics dictionaries into Pandas tables
    full_kin = {i+1: pd.DataFrame(d).stack() for i, d in enumerate(kinematics)}
    kinematics_pd = pd.concat(full_kin, axis=1, names=["index"]).swaplevel(0,1).T

    # Convert the central data values dict into Pandas tables
    f2pd = pd.DataFrame(f2_central, index=range(1, len(f2_central)+1), columns=["data"])
    f2pd.index.name = "index"
    f3pd = pd.DataFrame(f3_central, index=range(1, len(f3_central)+1), columns=["data"])
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


def extract_d2sigDxDy(path: Path, exp_name: str, table_id_list: list, obs: str) -> None:
    """
    Parameters:
    -----------
    path: Path
    """
    kinematics = []
    dsig_nu_central = []
    dsig_nu_errors = []
    console.print("\n• Extracting D2SIG/DX/DY from HEP tables:", style="bold blue")
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
                    "x": {"min": x_valmin, "mid": x_valmid, "max": x_valmax},
                    "Q2": {"min": q2_min, "mid": q2_mid, "max": q2_max},
                    "y": {"min": y_value_min, "mid": y_value_mid, "max": y_value_max}
                }
                kinematics.append(kin_dict)
                # ---- Extract central values for SF ---- #
                unc_type = dsig_dx_dy["values"][bin].get("error", None)
                if unc_type is None: stat_unc, syst_unc = None, None
                else:
                    stat_unc = unc_type[0].get("symerror", None)
                    syst_unc = unc_type[1].get("symerror", None)
                error_dict_1stentry = {"stat": stat_unc, "syst": syst_unc}
                dsig_nu_errors.append(error_dict_1stentry)

    # Convert the kinematics dictionaries into Pandas tables
    full_kin = {i+1: pd.DataFrame(d).stack() for i, d in enumerate(kinematics)}
    kinematics_pd = pd.concat(full_kin, axis=1, names=["index"]).swaplevel(0,1).T

    # Convert the central data values dict into Pandas tables
    nval_dnuu = len(dsig_nu_central) + 1
    dnuupd = pd.DataFrame(dsig_nu_central, index=range(1, nval_dnuu), columns=["data"])
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
    write_to_csv(systypes_folder, f"UNC_{exp_name}_DXDY{obs}", dsignuu_errors_pd)


if __name__ == "__main__":
    relative_path = Path().absolute().parents[0]
    experiment_name = "CDHSW"

    # List of tables containing measurements for F2 and xF3
    table_f3 = [i for i in range(19, 30)]
    extract_f2f3(relative_path, experiment_name, table_f3)

    # List of tables containing measurements for D2SIG/DX/DY for NUMU + FE
    table_dsig_dxdynuu = [i for i in range(1, 10)]
    extract_d2sigDxDy(relative_path, experiment_name, table_dsig_dxdynuu, "NUU")

    # List of tables containing measurements for D2SIG/DX/DY for NUMU + FE
    table_dsig_dxdynub = [i for i in range(10, 18)]
    extract_d2sigDxDy(relative_path, experiment_name, table_dsig_dxdynub, "NUB")
