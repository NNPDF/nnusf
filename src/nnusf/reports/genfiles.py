# -*- coding: utf-8 -*-
import json
import pathlib

import pandas as pd
import yaml

from ..plot.fit import prediction_data_comparison, training_validation_split


def dump_to_csv(
    fitfolder: pathlib.Path, pdtable: pd.DataFrame, filename: str
) -> None:
    """Dump a panda table into disk as csv."""
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/tables")
    output_path.mkdir(parents=True, exist_ok=True)
    pdtable.to_csv(f"{output_path}/{filename}.csv")


def json_loader(fitfolder: pathlib.Path) -> dict:
    with open(fitfolder, "r") as fstream:
        jsonfile = json.load(fstream)
    return jsonfile


def summary_table(fitfolder: pathlib.Path) -> pd.DataFrame:
    """Generate the table containing the summary of chi2s info.

    Parameters:
    -----------
        fitfolder: pathlib.Path
            Path to the fit folder
    """
    fitinfos = fitfolder.glob("**/replica_*/fitinfo.json")
    chi2_summary = {"tr": 0.0, "vl": 0.0, "Exp": 0.0}
    # Loop over the replica folder & extract chi2 info
    for count, repinfo in enumerate(fitinfos, start=1):
        jsonfile = json_loader(repinfo)
        for chi2type in ["tr", "vl"]:
            chi2_summary[chi2type] += jsonfile[f"best_{chi2type}_chi2"]
        chi2_summary["Exp"] += jsonfile["exp_chi2s"]["total_chi2"]

    # Average the chi2 over the nb of replicas
    for chi2type in chi2_summary:
        chi2_summary[chi2type] /= count
    chi2_summary["Exp"] /= count
    summtable = pd.DataFrame.from_dict({"chi2": chi2_summary})
    dump_to_csv(fitfolder, summtable, "summary")
    return summtable


def chi2_tables(fitfolder: pathlib.Path) -> pd.DataFrame:
    """Generate the table containing the chi2s info.

    Parameters:
    -----------
        fitfolder: pathlib.Path
            Path to the fit folder
    """
    # TODO: Add STDV to the averaged results
    runcard = fitfolder.joinpath("runcard.yml")
    runcard_content = yaml.load(runcard.read_text(), Loader=yaml.Loader)
    datinfo = runcard_content["experiments"]
    fitinfos = fitfolder.glob("**/replica_*/fitinfo.json")

    # Initialize dictionary to store the chi2 values
    dpts_dic = {d["dataset"]: d["frac"] for d in datinfo}
    chi2_dic = {
        d["dataset"]: {"Ndat": 0, "frac": 0, "tr_chi2": 0.0, "exp_chi2": 0.0}
        for d in datinfo
    }
    # Loop over the replica folder & extract chi2 info
    for count, repinfo in enumerate(fitinfos, start=1):
        jsonfile = json_loader(repinfo)
        for dat in chi2_dic:
            chi2_dic[dat]["Ndat"] = jsonfile["dtpts_per_dataset"][dat]
            chi2_dic[dat]["frac"] = dpts_dic[dat]
            chi2_dic[dat]["tr_chi2"] += jsonfile["chi2s_per_dataset"][dat]
            chi2_dic[dat]["exp_chi2"] += jsonfile["exp_chi2s"][dat]

    # Average the chi2 over the nb of replicas
    for dataset_name in chi2_dic:
        chi2_dic[dataset_name]["tr_chi2"] /= count
        chi2_dic[dataset_name]["exp_chi2"] /= count

    chi2table = pd.DataFrame.from_dict(chi2_dic, orient="index")
    dump_to_csv(fitfolder, chi2table, "chi2datasets")
    return chi2table


def data_vs_predictions(fitfolder: pathlib.Path) -> None:
    runcard = fitfolder.joinpath("runcard.yml")
    runcard_content = yaml.load(runcard.read_text(), Loader=yaml.Loader)

    # Prepare the output path to store the figures
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/figures")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the dictionary to pass to the action
    runcard_content["output"] = str(output_path)
    runcard_content["fit"] = str(fitfolder.absolute())

    prediction_data_comparison(**runcard_content)


def training_validation_plot(fitfolder: pathlib.Path) -> None:
    # Prepare the output path to store the figures
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/others")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the dictionary to pass to the action
    input_dic = {"fit": str(fitfolder.absolute()), "output": str(output_path)}

    training_validation_split(**input_dic)
