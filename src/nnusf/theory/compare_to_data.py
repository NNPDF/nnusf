# -*- coding: utf-8 -*-
"""Compute DIS predictions, out of given grids and compare to data."""
import logging
import pathlib
import tarfile
import tempfile
from typing import Optional

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.typing as npt
import pineappl

from .. import utils
from . import defs
from ..data import loader
from .predictions import theory_error, pdf_error
from .runcards import datsets_path

_logger = logging.getLogger(__name__)


def plot(
    pred: npt.NDArray,
    exp_data: pd.DataFrame,
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
        q2grid[: xpoint],
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

    plt.errorbar(
        q2grid[:, xpoint],
        exp_data['data'],
        yerr=np.sqrt(exp_data['stat']**2 + exp_data['syst']**2),
        color="tab:red",
        marker="x",
        label="data",
    )

    name, qualifier = obs.split("_")
    xpref = "x" if kind == "F3" else ""
    plt.title(f"${xpref}F_{{{name[1]},{qualifier}}}(x = {xgrid[0, xpoint]:.3g})$")
    plt.xscale("log")
    plt.legend()

    plt.savefig(preds_dest / f"{obs}.png")
    if interactive:
        plt.show()


Prediction = tuple[npt.NDArray[np.float_], int, slice, str]


def main(
    grids: pathlib.Path,
    dataset: pathlib.Path,
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
    datasets: pathlib.Path
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

        data_name = "_".join(dataset.stem.split("_")[1:])
        exp_data = loader.Loader(data_name, dataset.parents[1]).table
        xgrid, q2grid = np.meshgrid(*(exp_data['x'], exp_data['Q2']))
        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            obs = gpath.stem.split(".")[0]
            kind = obs.split("_")[0]

            grid = pineappl.grid.Grid.read(gpath)
            import pdb; pdb.set_trace()


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
                exp_data,
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

        tardest = destination / f"predictions-{data_name}.tar"
        with tarfile.open(tardest, "w") as tar:
            for path in preds_dest.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))

        _logger.info(f"Preedictions saved in '{tardest}'.")
