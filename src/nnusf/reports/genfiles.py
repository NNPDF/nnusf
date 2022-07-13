# -*- coding: utf-8 -*-
import json
import pathlib

import pandas as pd
import yaml

from ..plot.fit import prediction_data_comparison


def dump_to_csv(
    fitfolder: pathlib.Path, pdtable: pd.DataFrame, filename: str
) -> None:
    """Dump a panda table into disk as csv."""
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/tables")
    output_path.mkdir(parents=True, exist_ok=True)
    pdtable.to_csv(f"{output_path}/{filename}.csv")


def summary_table(fitfolder: pathlib.Path) -> pd.DataFrame:
    """Generate the table containing the summary of chi2s info.

    Parameters:
    -----------
        fitfolder: pathlib.Path
            Path to the fit folder
    """
    fitinfos = fitfolder.glob("**/replica_*/fitinfo.json")
    chi2_summary = {"tr": 0.0, "vl": 0.0}
    # Loop over the replica folder & extract chi2 info
    for count, repinfo in enumerate(fitinfos, start=1):
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        for chi2type in chi2_summary:
            chi2_summary[chi2type] += jsonfile[f"best_{chi2type}_chi2"]

    # Average the chi2 over the nb of replicas
    for chi2type in chi2_summary:
        chi2_summary[chi2type] /= count
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
    runcard_content = yaml.safe_load(runcard.read_text())
    datinfo = runcard_content["experiments"]
    fitinfos = fitfolder.glob("**/replica_*/fitinfo.json")

    # Initialize dictionary to store the chi2 values
    dpts_dic = {d["dataset"]: d["frac"] for d in datinfo}
    chi2_dic = {
        d["dataset"]: {"Ndat": 0, "frac": 0, "chi2": 0.0} for d in datinfo
    }
    # Loop over the replica folder & extract chi2 info
    for count, repinfo in enumerate(fitinfos, start=1):
        with open(repinfo, "r") as file_stream:
            jsonfile = json.load(file_stream)
        for dat in chi2_dic:
            chi2_dic[dat]["Ndat"] = jsonfile["dtpts_per_dataset"][dat]
            chi2_dic[dat]["frac"] = dpts_dic[dat]
            chi2_dic[dat]["chi2"] += jsonfile["chi2s_per_dataset"][dat]

    # Average the chi2 over the nb of replicas
    for dataset_name in chi2_dic:
        chi2_dic[dataset_name]["chi2"] /= count

    chi2table = pd.DataFrame.from_dict(chi2_dic, orient="index")
    dump_to_csv(fitfolder, chi2table, "chi2datasets")
    return chi2table


def data_vs_predictions(fitfolder: pathlib.Path) -> None:
    runcard = fitfolder.joinpath("runcard.yml")
    runcard_content = yaml.safe_load(runcard.read_text())
    datinfo = runcard_content["experiments"]
    dataset_list = [d["dataset"] for d in datinfo]

    # Prepare the output path to store the figures
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/figures")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the dictionary to pass to the action
    input_dic = {"experiments": dataset_list, "fit": "", "output": ""}
    input_dic["format"] = "png"
    input_dic["fit"] = str(fitfolder.absolute())
    input_dic["output"] = str(output_path)

    prediction_data_comparison(**input_dic)
