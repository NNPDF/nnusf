# -*- coding: utf-8 -*-
"""Generate heatmap plots for theory covariance matrix."""
import pathlib
from typing import Optional

import numpy as np

from .. import utils
from ..data import loader
from .covmat import (
    _EXP_ORD,
    construct_ticklabels,
    correlation_matrix,
    order_dict_experiment,
    save_heatmap,
)

_EXP_ORD.append("PROTONBC")


def compute(
    data: np.ndarray,
    central_values: np.ndarray,
    inverse: bool = False,
    norm: bool = True,
) -> np.ndarray:
    """Compute covmat.

    Parameters
    ----------
    array: np.ndarray
        array for which the covmat is computed
    central_values: np.ndarray
        array containing the central values
    inverse: bool
        if `True`, compute and plot the inverse of the covariance matrix
        (default: `False`)
    norm: bool
        if `True`, normalize the covariance matrix with central values (default:
        `True`)

    Returns
    -------
    np.ndarray
        (inverse) covariance matrix computed

    """
    covmat = np.cov(np.array(data).T)
    cv = np.array(central_values)
    if norm:
        covmat = covmat / cv / cv[:, np.newaxis]
    if inverse:
        covmat = np.linalg.inv(covmat)
    return covmat


def load_predictions(
    dataset_name: str,
    cpath: pathlib.Path,
    cuts: Optional[dict[str, dict[str, float]]] = None,
) -> np.ndarray:
    """Load theory predictions.

    Parameters
    ----------
    name: str
        name of the requested dataset
    cpath: pathlib.Path
        path to commondata
    cuts: dict
        kinematic cuts

    Returns
    -------
    (np.ndarray,np.ndarray)
        prediction for each replica
        theory shifts

    """
    sv_variations = []
    for variation in pathlib.Path(f"{cpath}/matching/").iterdir():
        if dataset_name in variation.stem:
            # Extract the central scale
            if "xif1.0_xir1.0" in variation.stem:
                nrep_predictions = np.load(variation)
            else:
                sv_variations.append(np.load(variation))
    # Build the shift in the theory predictions
    th_shift = sv_variations - nrep_predictions[:, 0]

    if cuts is not None:
        data = loader.Loader(dataset_name, cpath)
        mask_predictions = data.leftindex
        return nrep_predictions[mask_predictions].T, th_shift[mask_predictions]

    return nrep_predictions.T, th_shift


def main(
    data: list[pathlib.Path],
    destination: pathlib.Path,
    inverse: bool = False,
    norm: bool = True,
    cuts: Optional[dict[str, dict[str, float]]] = None,
):
    """Run covmat plot generation."""
    utils.mkdest(destination)

    normsuf = "" if not norm else "-norm"
    invsuf = "" if not inverse else "-inv"

    central_values = []
    pdf_pred = []
    th_shift = []
    predictions_dict = {}
    for ds in data:
        name, datapath = utils.split_data_path(ds)
        if "MATCHING" not in name:
            continue
        data_pred = load_predictions(name, datapath, cuts)
        predictions_dict[name] = data_pred[0][0]

        # Build the big matrices
        central_values.extend(data_pred[0][0])
        if th_shift == []:
            th_shift = data_pred[1].tolist()
            pdf_pred = data_pred[0].tolist()
            # here the pdf has just 33 replicas,
            # need to copy replica 0 to reach 200 replicas...
            if "BEBCWA59" in name:
                for i in range(33, 201):
                    pdf_pred.append(data_pred[0][0].tolist())
        else:
            rep_pred, th_pred = data_pred
            for i, var in enumerate(th_pred):
                th_shift[i].extend(var)
            for i, rep in enumerate(rep_pred):
                pdf_pred[i].extend(rep)
            if "BEBCWA59" in name:
                for i in range(33, 201):
                    pdf_pred[i].extend(data_pred[0][0].tolist())

    pdf_covmat = compute(pdf_pred, central_values, inverse, norm)
    th_covmat = compute(th_shift, central_values, inverse, norm)

    if inverse:
        # need to invert total only at the end
        pdf_covmat = compute(pdf_pred, central_values, inverse=False, norm=norm)
        th_covmat = np.cov(th_shift, central_values, inverse=False, norm=norm)
        total_covmat = th_covmat + pdf_covmat
        total_covmat = np.linalg.inv(total_covmat)
    else:
        total_covmat = th_covmat + pdf_covmat

    pdf_corr = correlation_matrix(pdf_covmat)
    th_corr = correlation_matrix(th_covmat)
    total_corr = correlation_matrix(total_covmat)

    # PDF matrices
    ordered_predictions = order_dict_experiment(predictions_dict, _EXP_ORD)
    ticklabels_specs = construct_ticklabels(ordered_predictions)
    save_heatmap(
        covmat=pdf_covmat,
        individual_data=False,
        figname=destination / f"pdf_covmat{normsuf}{invsuf}.pdf",
        title=r"$\rm{PDF~Covariance~Matrix~}$",
        tick_specs=ticklabels_specs,
    )
    save_heatmap(
        covmat=pdf_corr,
        individual_data=False,
        figname=destination / f"pdf_corrmat{normsuf}{invsuf}.pdf",
        title=r"$\rm{PDF~Correlation~Matrix~}$",
        tick_specs=ticklabels_specs,
    )

    # TH matrices
    save_heatmap(
        covmat=th_covmat,
        individual_data=False,
        figname=destination / f"th_covmat{normsuf}{invsuf}.pdf",
        title=r"$\rm{Theory~Covariance~Matrix~}$",
        tick_specs=ticklabels_specs,
    )
    save_heatmap(
        covmat=th_corr,
        individual_data=False,
        figname=destination / f"th_corrmat{normsuf}{invsuf}.pdf",
        title=r"$\rm{Theory~Correlation~Matrix~}$",
        tick_specs=ticklabels_specs,
    )

    # Total matrices
    save_heatmap(
        covmat=total_covmat,
        individual_data=False,
        figname=destination / f"total_covmat{normsuf}{invsuf}.pdf",
        title=r"$\rm{PDF+Theory~Covariance~Matrix~}$",
        tick_specs=ticklabels_specs,
    )
    save_heatmap(
        covmat=total_corr,
        individual_data=False,
        figname=destination / f"total_corrmat{normsuf}{invsuf}.pdf",
        title=r"$\rm{PDF+Theory~Correlation~Matrix~}$",
        tick_specs=ticklabels_specs,
    )
