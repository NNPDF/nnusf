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

    genie_pred = genie[f"{kind}_free_p"][gmask]
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
    __import__("pdb").set_trace()
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


def main(
    grids: pathlib.Path,
    pdf: str,
    destination: pathlib.Path,
    err: str = "theory",
    xpoint: Optional[int] = None,
    interactive: bool = False,
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

        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            obs = gpath.stem.split(".")[0]
            kind = obs.split("_")[0]

            grid = pineappl.grid.Grid.read(gpath)

            plt.figure()

            if err == "theory":
                pred, central, bulk, err_source = theory_error(
                    grid, pdf, defs.nine_points, xgrid
                )
            elif err == "pdf":
                pred, central, bulk, err_source = pdf_error(grid, pdf, xgrid)
            else:
                raise ValueError(f"Invalid error type '{err}'")

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

        tardest = destination / "predictions.tar"
        with tarfile.open(tardest, "w") as tar:
            for path in preds_dest.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))

        _logger.info(f"Preedictions saved in '{tardest}'.")
