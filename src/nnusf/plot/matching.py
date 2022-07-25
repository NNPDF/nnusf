# -*- coding: utf-8 -*-
"""Plot a dataset"""
import logging
import pathlib
import tarfile
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from .. import utils
from ..data import loader

_logger = logging.getLogger(__name__)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main(
    dataset: pathlib.Path,
    destination: pathlib.Path,
) -> None:
    """Plot a dataset

    Parameters
    ----------
    dataset: pathlib.Path
        path to matching dataset
    datasets: pathlib.Path
        path to
    """
    utils.mkdest(destination)

    data_name = "_".join(dataset.stem.split("_")[1:])
    exp_name = data_name.removesuffix("_MATCHING")
    data = loader.Loader(data_name, dataset.parents[1]).table
    data_exp = loader.Loader(exp_name, dataset.parents[1]).table

    temp_name = "MATCH_" + data_name.removesuffix("_MATCHING")
    nrep_predictions = np.load(f"{dataset.parents[1]}/matching/{temp_name}.npy")

    data["std"] = nrep_predictions.std(axis=1)
    # np.testing.assert_allclose(data["data"], nrep_predictions.mean(axis=1), atol=1e-6)
    # np.testing.assert_allclose(nrep_predictions.T[0], nrep_predictions.mean(axis=1), atol=1e-6)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        preds_dest = tmpdir / f"predictions-{data_name}"
        preds_dest.mkdir()

        # plot data with same x
        data = data.reset_index()
        for x in np.unique(data["x"]):

            x_exp = find_nearest(data_exp["x"], x)
            fixed_x_data_exp = data_exp[data_exp["x"] == x_exp]
            fixed_x_data = data[data["x"] == x]

            central = fixed_x_data["data"]
            central_exp = fixed_x_data_exp["data"]
            std = fixed_x_data["std"]
            std_exp = np.sqrt(
                fixed_x_data_exp["syst"] ** 2 + fixed_x_data_exp["stat"] ** 2
            )

            plt.figure()
            plt.errorbar(fixed_x_data["Q2"], central, yerr=std, fmt="o")
            plt.errorbar(
                fixed_x_data_exp["Q2"], central_exp, yerr=std_exp, fmt="x"
            )

            plt.title(f"{data_name}, x={x}. x_exp={x_exp:3f}")
            plt.xlabel("$Q^2$")
            plt.xscale("log")
            plt.tight_layout()
            plt.savefig(preds_dest / f"{data_name}_x_{x}.pdf")

        tardest = destination / f"predictions-{data_name}.tar"
        with tarfile.open(tardest, "w") as tar:
            for path in preds_dest.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))

        _logger.info(f"Plots saved in '{tardest}'.")
