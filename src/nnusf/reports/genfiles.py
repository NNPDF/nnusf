# -*- coding: utf-8 -*-
import json
import pathlib

import numpy as np
import pandas as pd
import yaml

from ..plot.fit import (
    prediction_data_comparison,
    training_epochs_distribution,
    training_validation_split,
)
from ..utils import compare_git_versions

MAP_LABELS = {
    "expr": r"\( \langle \chi^{2, \rm real}_{\rm exp} \rangle \)",
    "expt": r"\( \langle \chi^{2, \rm tot}_{\rm exp} \rangle \)",
    "tr": r"\( \langle \chi^{2}_{\rm tr} \rangle \)",
    "vl": r"\( \langle \chi^{2}_{\rm vl} \rangle \)",
}

COLUMN_LABELS = {
    "Ndat": r"\( \mathrm{N}_\mathrm{dat} \)",
    "frac": r"\( \mathrm{frac} \)",
    "tr_chi2": r"\( < \chi^{2, \star}_\mathrm{tr} > \)",
    "exp_chi2": r"\( < \chi^{2, \star}_{\mathrm{exp}} > \)",
}


def rename_dic_keys(curr_dic, new_keys):
    """Rename the keys of a dictionary."""

    for old_key, new_key in new_keys.items():
        curr_dic[new_key] = curr_dic.pop(old_key)


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
    summary = {}
    chi_tr, chi_vl, chi_real, chi_tot = [], [], [], []
    # Loop over the replica folder & extract chi2 info
    for repinfo in fitinfos:
        jsonfile = json_loader(repinfo)
        chi_tot.append(jsonfile["exp_chi2s"]["total_chi2"])
        chi_tr.append(jsonfile[f"best_tr_chi2"])
        chi_vl.append(jsonfile[f"best_vl_chi2"])
        chi_real.append(jsonfile["exp_chi2s"]["tot_chi2_real"])

    chi_real, chi_tot = np.asarray(chi_real), np.asarray(chi_tot)
    chi_tr, chi_vl = np.asarray(chi_tr), np.asarray(chi_vl)
    summary["tr"] = rf"{chi_tr.mean():.4f} \( \pm \) {chi_tr.std():.4f}"
    summary["vl"] = rf"{chi_vl.mean():.4f} \( \pm \) {chi_vl.std():.4f}"
    summary["expr"] = rf"{chi_real.mean():.4f} \( \pm \) {chi_real.std():.4f}"
    summary["expt"] = rf"{chi_tot.mean():.4f} \( \pm \) {chi_tot.std():.4f}"

    rename_dic_keys(summary, MAP_LABELS)
    summtable = pd.DataFrame.from_dict({"Values (STD)": summary})
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
    chi2table.rename(columns=COLUMN_LABELS, inplace=True)
    dump_to_csv(fitfolder, chi2table, "chi2datasets")
    return chi2table


def data_vs_predictions(fitfolder: pathlib.Path) -> None:
    runcard = fitfolder.joinpath("runcard.yml")
    runcard_content = yaml.load(runcard.read_text(), Loader=yaml.Loader)
    compare_git_versions(runcard_content)

    # Prepare the output path to store the figures
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/figures")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the dictionary to pass to the action
    runcard_content["output"] = str(output_path)
    runcard_content["fit"] = str(fitfolder.absolute())

    prediction_data_comparison(**runcard_content)


def additional_plots(fitfolder: pathlib.Path) -> None:
    # Prepare the output path to store the figures
    output_path = fitfolder.absolute()
    output_path = output_path.parents[0].joinpath("output/others")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the dictionary to pass to the action
    input_dic = {"fit": str(fitfolder.absolute()), "output": str(output_path)}

    training_validation_split(**input_dic)
    training_epochs_distribution(**input_dic)
