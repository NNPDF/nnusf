# -*- coding: utf-8 -*-
import functools
import logging

import h5py
import numpy as np

from ... import utils
from .cuts import q2cut, xcut

_logger = logging.getLogger(__name__)


class Data:
    def __init__(self, data: h5py.File):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    @property
    def members(self) -> list[str]:
        return list(self.data)

    def __getitem__(self, key) -> np.ndarray:
        return self.dataset(key)[:]

    def dataset(self, key: str) -> h5py.Dataset:
        if isinstance(key, int):
            ds = self.data[self.members[key]]
        else:
            ds = self.data[key]

        if not isinstance(ds, h5py.Dataset):
            raise ValueError(
                f"Only suitable for `Dataset`, while {key} is {type(ds)}"
            )

        return ds


@functools.cache
def load() -> Data:
    genie = h5py.File(utils.pkg / "theory" / "assets" / "genie.hdf5", "r")

    return Data(genie)


def kin_grids() -> tuple[np.ndarray, np.ndarray]:
    genie = load()

    xgrid = np.array(list(filter(xcut, genie["xlist"])))
    q2grid = np.array(list(filter(q2cut, genie["q2list"])))

    _logger.info(f"x: #{xgrid.size} {xgrid.min():4.3e} - {xgrid.max()}")
    _logger.info(
        f"Q: #{q2grid.size} {np.sqrt(q2grid.min()):4.3e} - {np.sqrt(q2grid.max()):4.3e}"
    )

    return xgrid, q2grid


def mask() -> np.ndarray:
    genie = load()

    xmask = xcut(genie["xlist"])
    q2mask = q2cut(genie["q2list"])

    return np.outer(xmask, q2mask)
