# -*- coding: utf-8 -*-
import lhapdf
import pathlib
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LHAPDF_ID = [1001, 2001, 3001, 1003, 2003, 3003]
XGRID = dict(min=1e-3, max=1.0, num=100)
SFS_LABEL = [
    r"$F_2^{\nu p} (x, Q^2)$",
    r"$F_2^{\bar{\nu} p} (x, Q^2)$",
    r"$F_2^{\left( \nu + \bar{\nu} \right) p} (x, Q^2)$",
    r"$x F_3^{\nu p} (x, Q^2)$",
    r"$x F_3^{\bar{\nu} p} (x, Q^2)$",
    r"$x F_3^{\left( \nu + \bar{\nu} \right) p} (x, Q^2)$",
]
MAP_XLABEL = {"x": r"$x$", "Q2": r"$Q^2$"}


class ValueNotAllowed(Exception):
    pass


def min_max_68_ci(data):
    """Compute the minimum and maximum values of the 68% confidence
    interval for a given array.

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    Tuple(float, float):
        The minimum and maximum values of the 68% confidence
        interval.

    """
    # Compute the mean and standard deviation of the data
    # Set ddof=1 to compute the sample standard deviation
    mean, std = np.mean(data), np.std(data, ddof=1)

    # Compute the 68% confidence interval using the z-score
    ci = mean + np.array([-1, 1]) * std / np.sqrt(len(data)) * 1.0
    return min(ci), max(ci)


def construct_lhapdf(
    pdfdict: dict, xgrid: np.ndarray, q2grid: np.ndarray
) -> pd.DataFrame:
    """Compute the LHAPDF predictions for all SF ID and a given
    kinematic points. The result is a Pandas Table containing all
    the relevant information.

    Parameters:
    -----------
    pdfdict: dict
        dictionary containing the SF sets with the labels
    xgrid: np.ndarray
        array of x values
    q2grid: np.ndarray
        array of Q2 values

    Returns:
    --------
    pd.DataFrame:
        pandas table with the predictions

    """
    # Initialize the LHAPDF set
    pdf = lhapdf.mkPDFs(pdfdict["pdfname"])

    prediction_table = [
        {
            "replica_id": rep_id,
            "x": x,
            "Q2": q2,
            "pid": id,
            "results": p.xfxQ2(id, x, q2),
        }
        for rep_id, p in enumerate(pdf)
        for id in LHAPDF_ID
        for x in xgrid
        for q2 in q2grid
    ]

    data_frame = pd.DataFrame(prediction_table)
    data_frame.insert(0, "label", pdfdict["label"])

    return data_frame


def compute_lhapdf(**kwargs) -> tuple[str, pd.DataFrame]:
    """Compute the LHAPDF predictions for all the SF sets requiered and
    concatenate the resulting predictions.

    Parameters:
    -----------
    kwargs: dict
        dictionary-like object containing the specs

    Returns:
    --------
    tuple(str, pd.DataFrame):
        tuple with the string specifying the dependent variable and the
        concatenated pandas table

    """
    if "fixed" in kwargs["x_values"] and "fixed" in kwargs["q2_values"]:
        raise ValueNotAllowed("Both x and Q2 cannot be fixed.")

    # Construct the kinematics with the corresponding grids
    if "fixed" not in kwargs["x_values"]:
        xgrid = np.logspace(
            np.log10(kwargs["x_values"].get("x_min", 1e-5)),
            np.log10(kwargs["x_values"].get("x_max", 1)),
            num=50,
        )
        dep_var, fixed_var = "varied_x", "fixed_Q2"
    else:
        xgrid = np.asarray([kwargs["x_values"].get("fixed", 1e-2)])

    if "fixed" not in kwargs["q2_values"]:
        q2grid = np.geomspace(
            kwargs["q2_values"].get("q2_min", 1e-1),
            kwargs["q2_values"].get("q2_max", 10),
            num=50,
        )
        dep_var, fixed_var = "varied_Q2", "fixed_x"
    else:
        q2grid = np.asarray([kwargs["q2_values"].get("fixed", 10)])

    pdfs = kwargs["sf_pdfs"]
    predictions = [construct_lhapdf(p, xgrid, q2grid) for p in pdfs]

    return dep_var, fixed_var, pd.concat(predictions, ignore_index=True)


def plot_sfs(df_pred: pd.DataFrame, dep_var: str, fixed_var) -> None:
    """Plot the (comparison of) Structure Functions as a function of
    the `dependent variable`.

    Parameters:
    -----------
    df_pred: pd.DataFrame
        pandas table containing the predictions
    dep_var: str
        string specifying the dependent variable
    fixed_var: str
        string specifying the fixed variable

    """
    fig, axs = plt.subplots(
        figsize=(3 * 5, 2 * 3.5),
        nrows=2,
        ncols=3,
        layout="constrained",
    )

    # Rename the `label` column for seaborn legend
    fixed_label = fixed_var.split("_")[-1]
    fixed_value = round(df_pred[fixed_label].iloc[0], 3)
    new_label = f"Fixed {fixed_label}={fixed_value}"
    df_pred = df_pred.rename(columns={"label": new_label})

    # Loop over axes OR Structure Function PIDs
    for id, ax in enumerate(axs.flat):
        # Subset of Data that matches a LHAPDF_ID
        new_df = df_pred.loc[df_pred["pid"] == LHAPDF_ID[id]]

        sns.lineplot(
            data=new_df,
            x=dep_var.split("_")[-1],
            y="results",
            hue=new_label,
            ax=ax,
            ci="sd",
        )

        # Define some specifictions for the plots
        ax.set_xlim(
            left=df_pred[dep_var.split("_")[-1]].min(),
            right=df_pred[dep_var.split("_")[-1]].max(),
        )
        ax.set_xscale("log")
        ax.set_xlabel(MAP_XLABEL[dep_var.split("_")[-1]])
        ax.set_ylabel(SFS_LABEL[id])

    fig.savefig(f"comparisons_sfs_{dep_var}.pdf", bbox_inches="tight", dpi=350)


def main(runcard: pathlib.Path, output: pathlib.Path):
    # Load the run card contaning all specs
    r_content = yaml.safe_load(runcard.read_text())
    depvar, fixvar, dfpred = compute_lhapdf(**r_content)

    # Plot & Save the structure function plots
    plot_sfs(df_pred=dfpred, dep_var=depvar, fixed_var=fixvar)
