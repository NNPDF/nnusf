# -*- coding: utf-8 -*-
"""Compute DIS predictions, out of given grids and compare to data."""
import logging
import pathlib
import tarfile
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pineappl

from .. import utils
from ..data import loader
from . import defs
from .predictions import pdf_error, theory_error

_logger = logging.getLogger(__name__)


def plot(
    pred: npt.NDArray,
    data: pd.DataFrame,
    data_name: str,
    x: float,
    central: int,
    bulk: slice,
    err_source: str,
    interactive: bool,
    preds_dest: pathlib.Path,
):
    """"""
    err = np.abs(pred[:, bulk].max(axis=1) - pred[:, bulk].min(axis=1)) / 2
    plt.errorbar(
        data["Q2"],
        pred[:, central],
        yerr=err,
        color="tab:blue",
        label=rf"yadism $\pm$ {err_source}",
        fmt="o",
        capsize=5,
    )

    plt.errorbar(
        data["Q2"],
        data["data"],
        yerr=np.sqrt(data["stat"] ** 2 + data["syst"] ** 2),
        color="tab:red",
        label=r"data $\pm$ (stat+syst)",
        fmt="x",
    )

    log_tab = {}
    log_tab["Q2"] = data["Q2"]
    log_tab["data"] = data["data"]
    log_tab["yadism"] = pred[:, central]
    log_tab["ratio"] = pred[:, central] / data["data"]
    _logger.info(f"x = {x}")
    _logger.info(pd.DataFrame(log_tab))

    plt.title(f"{data_name}, x={x}")
    plt.xlabel("$Q^2$")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()

    plt.savefig(preds_dest / f"{data_name}_x_{x}.pdf")
    if interactive:
        plt.show()


def main(
    grids: pathlib.Path,
    dataset: pathlib.Path,
    pdf: str,
    destination: pathlib.Path,
    err: str = "pdf",
    interactive: bool = False,
):
    """Run predictions computation.

    Parameters
    ----------
    grids: pathlib.Path
        path to grids archive
    datasets: pathlib.Path
        path to grids archive
    pdf: str
        LHAPDF name of the PDF to be used
    err: str
        type of error to be used

    """
    utils.mkdest(destination)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        # extract tar content
        if grids.suffix == ".tar":
            utils.extract_tar(grids, tmpdir)
            grids = tmpdir / "grids"

        data_name = "_".join(dataset.stem.split("_")[1:])
        preds_dest = tmpdir / f"predictions-{data_name}"
        preds_dest.mkdir()

        data = loader.Loader(data_name, dataset.parents[1]).table
        xgrid, q2grid = data["x"], data["Q2"]

        # do the average if necessary
        full_pred = []
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            grid = pineappl.grid.Grid.read(gpath)
            if err == "theory":
                pred, central, bulk, err_source = theory_error(
                    grid, pdf, defs.nine_points, xgrid, reshape=False
                )
            elif err == "pdf":
                pred, central, bulk, err_source = pdf_error(
                    grid, pdf, xgrid, reshape=False
                )
            else:
                raise ValueError(f"Invalid error type '{err}'")
            full_pred.append(pred)
        pred = np.average(full_pred, axis=0)

        # save predictions
        np.save(preds_dest / data_name, pred)
        np.save(preds_dest / "xgrid", xgrid)
        np.save(preds_dest / "q2grid", q2grid)

        # plot data with same x
        data = data.reset_index()
        for x in np.unique(xgrid):
            plt.figure()
            idx = data[data["x"] == x].index
            plot(
                pred[idx, :],
                data[data["x"] == x],
                data_name,
                x,
                central,
                bulk,
                err_source,
                interactive,
                preds_dest,
            )

        tardest = destination / f"predictions-{data_name}.tar"
        with tarfile.open(tardest, "w") as tar:
            for path in preds_dest.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))

        _logger.info(f"Preedictions saved in '{tardest}'.")
