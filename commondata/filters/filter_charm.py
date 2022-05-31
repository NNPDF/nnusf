#!/usr/bin/env python3

import yaml
import pandas as pd

from pathlib import Path
from rich.console import Console
from rich.progress import track

ERR_DESC = {
    'syst': {
        'treatment': "ADD",
        'type': "CORR",
        'description': "Total systematic uncertainty"
    }
}

console = Console()


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
        indep_var_dic = load_table["independent_variables"]
        # Loop over the pair of (F2, xF3) observables which are alternated
        dependent_variables = load_table["dependent_variables"]
        nb_pairs_f2f3_loops = len(dependent_variables)
        for i in range(0, nb_pairs_f2f3_loops, 2):
            dep_var_f2dic = load_table["dependent_variables"][i]   # F2
            dep_var_f3dic = load_table["dependent_variables"][i+1] # xF3
            # The x values should be the same for F2 & xF3
            f2_x_value = float(dep_var_f2dic["qualifiers"][3]["value"])
            f3_x_value = float(dep_var_f3dic["qualifiers"][3]["value"])
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
                        syst_unc = None
                    else:
                        syst_unc = unc_type[0].get("symerror", None)
                    uncertainty_dic[uncertainty_names[idx]] = syst_unc
                error_dict_f2 = {
                    "syst": uncertainty_dic["f2_unc"]
                }
                f2_exp_errors.append(error_dict_f2)
                error_dict_f3 = {
                    "syst": uncertainty_dic["f3_unc"]
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


if __name__ == "__main__":
    relative_path = Path().absolute().parents[0]
    experiment_name = "CHARM"

    # List of tables containing measurements for F2 and xF3
    extract_f2f3(relative_path, experiment_name, [1])
