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

from .. import utils
from . import defs
from .bodek_yang import load

_logger = logging.getLogger(__name__)


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
        pred = np.array(pred).T.reshape((*xgrid.shape, len(pred)))
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
    return pred, 0, slice(1, -1), "PDF replicas"


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
    reshape: bool = True,
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
        xgrid, q2grid = np.meshgrid(*load.kin_grids())
        gmask = load.mask()

        combined_predictions = []
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            obs = gpath.stem.split(".")[0]
            kind = obs.split("_")[0]

            grid = pineappl.grid.Grid.read(gpath)

            plt.figure()

            if err == "theory":
                pred, central, bulk, err_source = theory_error(
                    grid, pdf, defs.nine_points, xgrid, reshape
                )
            elif err == "pdf":
                pred, central, bulk, err_source = pdf_error(
                    grid, pdf, xgrid, reshape
                )
            else:
                raise ValueError(f"Invalid error type '{err}'")

            if reshape:
                plot(
                    pred,
                    genie,
                    gmask,
                    obs,
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
            else:
                gridname = gpath.stem.split(".")[0]
                nuc = gridname.split("-")[0].split("_")[-1]
                sf_type = gridname.split("-")[-1]
                neutrino_type = gridname.split("_")[0]
                if sf_type == "F3":
                    obsname = "x" + sf_type + neutrino_type
                else:
                    obsname = sf_type + neutrino_type
                dict_pred = {
                    "nucleus": nuc,
                    "predictions": np.expand_dims(np.asarray(pred), axis=0),
                    "obsname": obsname,
                }
                combined_predictions.append(dict_pred)

        if reshape:
            tardest = destination / "predictions.tar"
            with tarfile.open(tardest, "w") as tar:
                for path in preds_dest.iterdir():
                    tar.add(path.absolute(), path.relative_to(tmpdir))

            _logger.info(f"Preedictions saved in '{tardest}'.")
        else:
            generate_txt(combined_predictions)
