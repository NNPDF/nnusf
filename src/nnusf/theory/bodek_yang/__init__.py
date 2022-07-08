# -*- coding: utf-8 -*-
import logging
import pathlib

import numpy as np

from ... import utils
from . import load, runcards

_logger = logging.getLogger(__name__)


def extract(observables: list[str]) -> tuple[np.ndarray, list[str]]:
    data = load.load()

    xgrid = data["xlist"]
    q2grid = data["q2list"]

    shape = (xgrid.size, q2grid.size)

    labels = ["x", "Q2"]
    values = [
        np.broadcast_to(xgrid[:, np.newaxis], shape),
        np.broadcast_to(q2grid, shape),
    ]
    for obs in observables:
        labels.append(obs)
        values.append(data[obs])

    return np.stack(values), labels


def dump_text(ar: np.ndarray, labels: list[str], destination: pathlib.Path):
    table = ar.reshape(ar.shape[0], np.prod(ar.shape[1:]))
    fname = destination / "genie-extracted.txt"

    utils.mkdest(destination)

    np.savetxt(fname, table.T, header=" ".join(labels))
    _logger.info(
        f"Saved Genie data in '{fname.relative_to(pathlib.Path.cwd())}'"
    )


def dump(ar: np.ndarray, destination: pathlib.Path):
    fname = destination / "genie-extracted.npy"

    utils.mkdest(destination)

    np.save(fname, ar)
    _logger.info(
        f"Saved Genie data in '{fname.relative_to(pathlib.Path.cwd())}'"
    )
