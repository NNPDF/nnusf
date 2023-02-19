# -*- coding: utf-8 -*-
import logging
import pathlib

import numpy as np
from yadism.esf.exs import xs_coeffs

from ..theory import runcards
from . import loader, utils

_logger = logging.getLogger(__name__)


class MissingRequisite(ValueError):
    pass


def cross_section(
    name: str, kinematics: np.ndarray, y: np.ndarray, proj: int, pos: np.ndarray
):
    exp = name.split("_")[0]
    try:
        xs = utils.MAP_EXP_YADISM[exp]
        if name == "CDHSW_FW":
            xs = "FW"
    except KeyError:
        raise MissingRequisite(f"NO available cross-section for '{exp}'")

    yadcoeffs = []

    for kin, y in zip(kinematics, y):
        th = runcards.theory()
        yadcoeffs.append(
            xs_coeffs(
                xs,
                y=y,
                x=kin[0],
                Q2=kin[1],
                params=dict(
                    projectilePID=proj,
                    M2target=th["MP"] ** 2,
                    M2W=th["MW"] ** 2,
                    GF=th["GF"],
                ),
            )
        )

    yadcoeffs = np.array(yadcoeffs)
    idx = np.sum(np.meshgrid(np.arange(3), pos), axis=0).flatten()

    coeffs = np.zeros((len(kinematics), 6))
    coeffs[:, idx] = np.transpose(
        yadcoeffs[:, :, np.newaxis]
        * np.ones_like(pos)[np.newaxis, np.newaxis, :],
        [0, 2, 1],
    ).reshape(yadcoeffs.shape[0], yadcoeffs.shape[1] * pos.size)

    return coeffs


def coefficients(name: str, datapath: pathlib.Path):
    data = loader.Loader(name, datapath)

    if data.n_data == 0:
        MissingRequisite("NO data found")

    proj = data.table["projectile"].values[0]

    if proj < 0:
        pos = [3]
    elif proj == 0:
        pos = [0, 3]
    elif proj > 0:
        pos = [0]
    else:
        raise ValueError
    pos = np.array(pos)

    if data.obs in ["F2", "F3"]:
        coeffs = np.zeros((data.n_data, 6))
        pos += 0 if data.obs == "F2" else 2
        coeffs[:, pos] = 1
    else:
        if "y" not in data.table:
            raise MissingRequisite(f"NO y available for '{name}'")

        coeffs = cross_section(
            name,
            kinematics=data.kinematics,
            y=data.table["y"].values,
            proj=proj,
            pos=pos,
        )

    # if average, divide by factor
    coeffs[np.sign(proj) == 0] *= 1.0 / 2.0

    return coeffs


def main(datapaths: list[pathlib.Path], destination: pathlib.Path):
    destination.mkdir(parents=True, exist_ok=True)

    for dataset in datapaths:
        name = dataset.stem.strip("DATA_")

        try:
            coeffs = coefficients(name, datapath=dataset.parents[1])
        except MissingRequisite as exc:
            _logger.error(f"{exc.args[0]}")
            continue

        dest = (destination / name).with_suffix(".npy")
        np.save(dest, coeffs)
        _logger.info(
            f"The coefficient with shape {coeffs.shape} is saved "
            f"in {dest.absolute().relative_to(pathlib.Path.cwd())}"
        )
