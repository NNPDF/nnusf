# -*- coding: utf-8 -*-
import json
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from ..plot.fit import (
    prediction_data_comparison,
    training_epochs_distribution,
    training_validation_split,
)
from ..sffit.load_data import load_experimental_data
from ..sffit.load_fit_data import load_models
from ..utils import compare_git_versions

MAP_LABELS = {
    "chi2tot": r"\( \chi^{2, \rm real}_{\rm total} \)",
    "expr": r"\( \langle \chi^{2, \rm real}_{\mathrm{exp}, k} \rangle \)",
    "expt": r"\( \langle \chi^{2, \rm tot}_{\mathrm{exp}, k} \rangle \)",
    "tr": r"\( \langle \chi^{2}_{\rm tr} \rangle \)",
    "vl": r"\( \langle \chi^{2}_{\rm vl} \rangle \)",
}

COLUMN_LABELS = {
    "tot_chi2": r"\( \chi^{2}_\mathrm{total} \)",
    "Ndat": r"\( \mathrm{N}_\mathrm{dat} \)",
    "frac": r"\( \mathrm{frac} \)",
    "tr_chi2": r"\( < \chi^{2, k}_\mathrm{tr} > \)",
    "exp_chi2": r"\( < \chi^{2, k}_{\mathrm{exp}} > \)",
}

NFIT_LABELS = {"nonfit_chi2": r"\( < \chi^{2, \star}_{\mathrm{exp}} > \)"}


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
    """Load a JSON file."""
    with open(fitfolder, "r") as fstream:
        jsonfile = json.load(fstream)
    return jsonfile


def addinfo_yaml(fitfolder: pathlib.Path) -> dict:
    """Add required info to run the saved models."""
    runcard = fitfolder.joinpath("runcard.yml")
    runcard_content = yaml.load(runcard.read_text(), Loader=yaml.SafeLoader)
    compare_git_versions(runcard_content)
    runcard_content["fit"] = str(fitfolder.absolute())
    return runcard_content


def _compute_chi2(expdata, fitpred, invcovmat) -> np.ndarray:
    """Compute the Chi2 given the experimental data,
    the fitted predictions, and the inverse covmats.

    Parameters:
    -----------
    expdata: np.ndarray
        array of experimental data
    fitpred; np.array
        array of NN fitted predictions
    invcovmat: np.ndarray
        array of covariance matrix

    Returns:
    --------
    float:
        value of the total chi2
    """
    diff_predictions = expdata - fitpred
    right_dot = np.tensordot(invcovmat, np.transpose(diff_predictions), axes=1)
    return np.tensordot(diff_predictions, right_dot, axes=1)


def build_data_models(**kwargs) -> tuple[list, dict, dict]:
    """Build the data object and load the trained models."""
    models = load_models(**kwargs)

    # Load the datasets all at once in order to rescale
    raw_datasets, datasets = load_experimental_data(
        kwargs["experiments"],
        input_scaling=kwargs.get("rescale_inputs", None),
        kincuts=kwargs.get("kinematic_cuts", {}),
        verbose=False,
    )

    return models, raw_datasets, datasets


def compute_totchi2(models: list, datasets: dict) -> tuple[float, dict]:
    """Compute the total chi2 value. The total Chi2 is computed
    by comparing the mean of the NN predictions with the central
    data values.
    """

    # Compute the total Chi2 per dataset
    totchi2_dataset = {"normalized": {}, "unnormalized": {}, "ndata": {}}
    for _, data in datasets.items():
        # NOTE: The kinematics below are scaled
        kins = np.expand_dims(data.kinematics, axis=0)
        # Loop over the Neural Network models
        obs = []
        for model in models:
            prediction = model(kins)
            prediction = prediction[0]  # remove batch dimension
            obs.append(tf.einsum("ij,ij->i", prediction, data.coefficients))
        obs_mean = np.asarray(obs).mean(axis=0)

        # Compute the Inverse of the Covmat
        invcovmat = np.linalg.inv(data.covmat)

        # Compute the value of the Chi2
        chi2tot = _compute_chi2(
            expdata=data.central_values,
            invcovmat=invcovmat,
            fitpred=obs_mean,
        )
        totchi2_dataset["unnormalized"][data.name] = float(chi2tot)
        totchi2_dataset["ndata"][data.name] = kins.shape[1]
        totchi2_dataset["normalized"][data.name] = chi2tot / kins.shape[1]

    # Compute total Chi2 for experimental datasets
    totchi2_exp = sum(
        [
            v
            for k, v in totchi2_dataset["unnormalized"].items()
            if "_MATCHING" not in k
        ]
    )
    totchi2_exp /= sum(
        [v for k, v in totchi2_dataset["ndata"].items() if "_MATCHING" not in k]
    )

    return totchi2_exp, totchi2_dataset["normalized"]


def nonfitted_chi2(
    models: list,
    max_kins: np.ndarray,
    fitfolder: pathlib.Path,
    **kwargs,
) -> pd.DataFrame:
    """Compute the Chi2 of the datasets that were not included in
    the fit using the saved pre-trained models."""

    _, datasets = load_experimental_data(
        kwargs["check_chi2_experiments"],
        input_scaling=kwargs.get("rescale_inputs", None),
        kincuts=kwargs.get("kinematic_cuts", {}),
        verbose=False,
        max_kin=max_kins,
    )

    _, nonfitted_chi2 = compute_totchi2(models, datasets)

    chi2s = {n: {"nonfit_chi2": v} for n, v in nonfitted_chi2.items()}

    chi2table = pd.DataFrame.from_dict(chi2s, orient="index")
    chi2table.rename(columns=NFIT_LABELS, inplace=True)
    dump_to_csv(fitfolder, chi2table, "nonfitted_chi2datasets")
    return chi2table


def summary_table(fitfolder: pathlib.Path, chi2tot: float) -> pd.DataFrame:
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

    summary["chi2tot"] = rf"{chi2tot:.4f}"
    summary["tr"] = rf"{chi_tr.mean():.4f} \( \pm \) {chi_tr.std():.4f}"
    summary["vl"] = rf"{chi_vl.mean():.4f} \( \pm \) {chi_vl.std():.4f}"
    summary["expr"] = rf"{chi_real.mean():.4f} \( \pm \) {chi_real.std():.4f}"
    summary["expt"] = rf"{chi_tot.mean():.4f} \( \pm \) {chi_tot.std():.4f}"

    # TODO: Use pd.rename(columns=MAPPING, inplace=True) instead
    rename_dic_keys(summary, MAP_LABELS)
    summtable = pd.DataFrame.from_dict({"Values (STD)": summary})
    dump_to_csv(fitfolder, summtable, "summary")
    return summtable


def chi2_tables(fitfolder: pathlib.Path, chi2_datasets: dict) -> pd.DataFrame:
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
        chi2_dic[dataset_name]["tot_chi2"] = chi2_datasets[dataset_name]

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
