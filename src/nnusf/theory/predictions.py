# -*- coding: utf-8 -*-
"""Compute DIS predictions, out of given grids."""
import logging
import pathlib
import tarfile
import tempfile
from typing import Optional

import lhapdf
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pineappl
import yaml

from . import defs
from .bodek_yang import load
from .. import utils
from ..lhapdf.dump_grids import LHAPDF_ID, dump_pred_lhapdf
from ..utils import ROUNDING

_logger = logging.getLogger(__name__)

# The labels and IDs elow have to match exactly
SFS_LABEL = ["F2nu", "FLnu", "xF3nu", "F2nub", "FLnub", "xF3nub"]


def plot(
    pred: npt.NDArray,
    genie: load.Data,
    gmask: npt.NDArray,
    obs: str,
    kind: str,
    xgrid: npt.NDArray,
    q2grid: npt.NDArray,
    xpoint: int,
    central: int,
    bulk: slice,
    err_source: str,
    interactive: bool,
    preds_dest: pathlib.Path,
):
    """"""
    plt.figure()
    plt.plot(
        q2grid[:, xpoint],
        pred[:, xpoint, central],
        color="tab:blue",
        label="yadism",
    )
    plt.fill_between(
        q2grid[:, xpoint],
        pred[:, xpoint, bulk].min(axis=1),
        pred[:, xpoint, bulk].max(axis=1),
        facecolor=clr.to_rgba("tab:blue", alpha=0.1),
        label=err_source,
    )
    np.save(preds_dest / obs, pred)
    np.save(preds_dest / "xgrid", xgrid)
    np.save(preds_dest / "q2grid", q2grid)

    try:
        genie_pred = genie[f"{kind}_free_p"][gmask]
    except KeyError:
        # if there are no Genie predictions, plot nothing
        return
    if kind == "F3":
        genie_pred = xgrid.T.flatten() * genie_pred

    genie_pred = genie_pred.reshape(tuple(reversed(xgrid.shape)))
    plt.scatter(
        q2grid[:, xpoint],
        genie_pred[xpoint],
        color="tab:red",
        marker="x",
        label="genie",
    )

    name, qualifier = obs.split("_")
    xpref = "x" if kind == "F3" else ""
    plt.title(
        f"${xpref}F_{{{name[1]},{qualifier}}}(x = {xgrid[0, xpoint]:.3g})$"
    )
    plt.xscale("log")
    plt.legend()

    plt.savefig(preds_dest / f"{obs}.png")
    if interactive:
        plt.show()


Prediction = tuple[npt.NDArray[np.float_], int, slice, str]


def theory_error(
    grid: pineappl.grid.Grid,
    pdf: str,
    prescription: list[tuple[float, float]],
    xgrid: npt.NDArray[np.float_],
    reshape: Optional[bool] = True,
) -> Prediction:
    # theory uncertainties
    pdfset = lhapdf.mkPDF(pdf)
    pred = grid.convolute_with_one(
        2212, pdfset.xfxQ2, pdfset.alphasQ2, xi=prescription
    )

    if reshape:
        pred = np.array(pred).T.reshape((*xgrid.shape, 9))
    else:
        pred = np.array(pred).T
    return pred, 4, slice(0, -1), "9 pts."


def pdf_error(
    grid: pineappl.grid.Grid,
    pdf: str,
    xgrid: npt.NDArray[np.float_],
    reshape: Optional[bool] = True,
) -> Prediction:
    """Compute PDF uncertainties"""
    pred = []
    for pdfset in lhapdf.mkPDFs(pdf):
        member_pred = grid.convolute_with_one(
            2212, pdfset.xfxQ2, pdfset.alphasQ2
        )
        pred.append(member_pred)

    if reshape:
        pred = np.array(pred).T.reshape((*xgrid.shape, len(pred)))
    else:
        pred = np.array(pred).T
    # Make sure that `pred` does not include Member 0
    return pred[:, :, 1:], 0, slice(1, -1), "PDF replicas"


def combined_error(
    grid: pineappl.grid.Grid,
    pdf: str,
    prescription: list[tuple[float, float]],
    xgrid: npt.NDArray[np.float_],
    reshape: Optional[bool] = True,
) -> npt.NDArray[np.float_]:
    # TODO: To implement the combination
    pred_theory, _, _, _ = pdf_error(
        grid,
        pdf,
        xgrid,
        reshape,
    )
    return pred_theory


def construct_stacked_predictions(predictions_dict: dict):
    """Stack the predictions in the same order as the LHAPDF IDs."""
    stacked_list = [predictions_dict[k] for k in SFS_LABEL]
    # Stack the predictions such that the shape now becomes
    # (n_x, n_q2, n_rep, n_sfs)
    stacked_predictions = np.stack(stacked_list, axis=-1)
    # Move axis to get shape (n_rep, n_q2, n_x, n_sfs)
    return np.moveaxis(stacked_predictions, [0, 2], [2, 0])


def extract_xq2_from_pineappl(grid: pineappl.grid.Grid, obsname: str):
    """Extract the unique x and Q2 values from the grid."""
    cards = yaml.safe_load(grid.raw.key_values()["runcard"])
    observables = cards["observables"][obsname]

    x_grids = np.asarray([float(v["x"]) for v in observables])
    q2grid = np.asarray([float(v["Q2"]) for v in observables])

    # Extract only the unique kinematic values
    unique_xv, unique_q2 = np.unique(x_grids), np.unique(q2grid)
    # Make sure that the unique values indeed is consistent
    assert (unique_xv.size * unique_q2.size) == len(observables)

    return unique_xv, unique_q2


def parse_yadism_predictions(
    predictions: np.ndarray,
    x_specs: np.ndarray,
    q2_dic_specs: np.ndarray,
):
    prediction_infoq2 = [round(q, ROUNDING) for q in q2_dic_specs]

    # Compute the central replicas and prepend to array
    central = np.expand_dims(np.mean(predictions, axis=0), axis=0)
    predictions = np.concatenate([central, predictions], axis=0)

    # Append the average to the array block
    copied_pred = np.copy(predictions)
    for i in range(copied_pred.shape[-1] // 2):
        avg = (copied_pred[:, :, :, i] + copied_pred[:, :, :, i + 3]) / 2
        average = np.expand_dims(avg, axis=-1)
        predictions = np.concatenate([predictions, average], axis=-1)

    # Parse the array blocks as a Dictionary
    combined_replica = []
    for replica in predictions:  # loop over the replica
        q2rep_dic = {}
        # Loop over the results for all Q2 values
        for idq, q2rep in enumerate(replica):
            sfs_q2rep = {}
            q2rep_idx = prediction_infoq2[idq]
            # loop over the Structure Functions
            for idx in range(q2rep.shape[-1]):
                sfs_q2rep[LHAPDF_ID[idx]] = q2rep[:, idx]
            q2rep_dic[round(q2rep_idx, ROUNDING)] = sfs_q2rep
        combined_replica.append(q2rep_dic)

    grids_info_specs = {
        "x_grids": x_specs.tolist(),
        "q2_grids": prediction_infoq2,
        "nrep": len(combined_replica),
    }
    return grids_info_specs, combined_replica


def generate_txt(predictions: list[dict]) -> None:
    pred_label = []
    pred_results = []
    nuc_types = []
    for pred in predictions:
        pred_label.append(pred["obsname"])
        pred_results.append(pred["predictions"])
        nuc_types.append(pred["nucleus"])
    nuc_info = [i for i in list(set(nuc_types))]
    assert len(nuc_info) == 1

    combined_pred = np.concatenate(pred_results)
    combined_pred = np.moveaxis(combined_pred, [0, 1, 2], [2, 1, 0])

    xval = [0.1]
    q2_grids = np.geomspace(5, 1e3, num=400)

    stacked_results = []
    for idx, pr in enumerate([combined_pred]):
        predshape = pr[:, :, 0].shape
        broad_xvalues = np.broadcast_to(xval[idx], predshape)
        broad_qvalues = np.broadcast_to(q2_grids, predshape)
        # Construct the replica index array
        repindex = np.arange(pr.shape[0])[:, np.newaxis]
        repindex = np.broadcast_to(repindex, predshape)
        # Stack all the arrays together
        stacked_list = [repindex, broad_xvalues, broad_qvalues]
        stacked_list += [pr[:, :, i] for i in range(pr.shape[-1])]
        stacked = np.stack(stacked_list).reshape((9, -1)).T
        stacked_results.append(stacked)
    final_predictions = np.concatenate(stacked_results, axis=0)
    header = " "
    for sflabel in pred_label:
        header = header + sflabel + " "
    np.savetxt(
        f"yadism_sfs_{nuc_info[0]}.txt",
        final_predictions,
        header=f"replica x Q2" + header,
        fmt="%d %e %e %e %e %e %e %e %e",
    )


def main(
    grids: pathlib.Path,
    pdf: str,
    destination: pathlib.Path,
    err: str = "pdf",
    xpoint: Optional[int] = None,
    interactive: bool = False,
    compare_to_by: bool = True,
):
    """Run predictions computation.

    Parameters
    ----------
    grids: pathlib.Path
        path to grids archive
    pdf: str
        LHAPDF name of the PDF to be used
    err: str
        type of error to be used
    xpoint: int or None
        point in Bjorken x to be used for the slice to plot

    """
    utils.mkdest(destination)

    if xpoint is None:
        xpoint = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        # extract tar content
        if grids.suffix == ".tar":
            utils.extract_tar(grids, tmpdir)
            grids = tmpdir / "grids"

        preds_dest = tmpdir / "predictions"
        preds_dest.mkdir()

        genie = load.load()
        gmask = load.mask()

        predictions_dictionary = {}
        nucleus_predictions = []
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            gridname = gpath.stem.split(".")[0]
            kind = gridname.split("_")[0]

            grid = pineappl.grid.Grid.read(gpath)

            # Extract the information on the input kinematics
            if compare_to_by:
                xgrid, q2grid = np.meshgrid(*load.kin_grids())
            else:
                sf_type = gridname.split("-")[-1]
                x_grid, q2_grid = extract_xq2_from_pineappl(grid, sf_type)
                xgrid, q2grid = np.meshgrid(*tuple((q2_grid, x_grid)))

            if err == "theory":
                pred, central, bulk, err_source = theory_error(
                    grid,
                    pdf,
                    defs.nine_points,
                    xgrid,
                )
            elif err == "pdf":
                pred, central, bulk, err_source = pdf_error(
                    grid,
                    pdf,
                    xgrid,
                )
            elif err == "combined":
                pred, central, bulk, err_source = combined_error(
                    grid,
                    pdf,
                    defs.nine_points,
                    xgrid,
                )
            else:
                raise ValueError(f"Invalid error type '{err}'")

            # TODO: Remove the below as the comparisons are done
            # usually done outside of this module.
            if compare_to_by:  # Compare Yadism with BY
                plot(
                    pred,
                    genie,
                    gmask,
                    gridname,
                    kind,
                    xgrid,
                    q2grid,
                    xpoint,
                    central,
                    bulk,
                    err_source,
                    interactive,
                    preds_dest,
                )
            else:  # Dump the Yadism results as LHAPDF
                nuc = gridname.split("-")[0].split("_")[-1]
                neutrino_type = gridname.split("_")[0]
                if sf_type == "F3":
                    obsname = "x" + sf_type + neutrino_type
                else:
                    obsname = sf_type + neutrino_type
                nucleus_predictions.append(nuc)
                predictions_dictionary[obsname] = pred

        if compare_to_by:
            tardest = destination / "predictions.tar"
            with tarfile.open(tardest, "w") as tar:
                for path in preds_dest.iterdir():
                    tar.add(path.absolute(), path.relative_to(tmpdir))

            _logger.info(f"Preedictions saved in '{tardest}'.")
        else:
            nuc_info = [i for i in list(set(nucleus_predictions))]
            assert len(nuc_info) == 1
            stacked_pred = construct_stacked_predictions(predictions_dictionary)
            grid_info, cpred = parse_yadism_predictions(
                stacked_pred, x_grid, q2_grid
            )
            dump_pred_lhapdf(
                f"YADISM_{nuc_info[0]}",
                nuc_info[0],
                cpred,
                grid_info,
                LHAPDF_ID,
            )
