# -*- coding: utf-8 -*-
"""
Module that the Structure Function predictions from the NN and
dump them as a LHAPDF-like grid.
"""

import logging
import pathlib
from typing import Union

import numpy as np
from rich.progress import track

from ..sffit.load_fit_data import get_predictions_q
from .utils import create_info_file, dump_set, generate_block, install_pdf

_logger = logging.getLogger(__name__)

LHAPDF_ID = [1001, 1002, 1003, 2001, 2002, 2003, 3001, 3002, 3003]
A_VALUE = 1
X_GRIDS = dict(min=1e-2, max=1.0, num=100)
Q2_GRIDS = dict(min=1, max=500, num=400)


def parse_nn_predictions(
    model: pathlib.Path, a_value_spec: int, x_specs: dict, q2_dic_specs: dict
):
    x_grids = np.linspace(x_specs["min"], x_specs["max"], x_specs["num"])
    prediction_info = get_predictions_q(
        fit=model,
        a_slice=a_value_spec,
        x_slice=x_grids.tolist(),
        qmin=q2_dic_specs["min"],
        qmax=q2_dic_specs["max"],
        n=q2_dic_specs["num"],
    )
    prediction_infoq2 = [round(q, 3) for q in prediction_info.q]
    predictions = np.asarray(prediction_info.predictions)
    # The predictions above is of shape (nx, nrep, n_q2, n_sfs)
    # and the moveaxis transforms it into (nrep, n_q2, n_x, n_sfs)
    predictions = np.moveaxis(predictions, [0, 1, 2], [2, 0, 1])

    # Append the average to the array block
    for i in range(0, predictions.shape[-1], 2):
        avg = (predictions[:, :, :, i] + predictions[:, :, :, i + 1]) / 2
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
            q2rep_dic[round(q2rep_idx, 3)] = sfs_q2rep
        combined_replica.append(q2rep_dic)

    grids_info_specs = {
        "x_grids": prediction_info.x,
        "q2_grids": prediction_infoq2,
        "nrep": len(combined_replica),
    }

    return grids_info_specs, combined_replica


def dump_pred_lhapdf(name: str, all_replicas: list, grids_info_specs: dict):
    # Generate the dictionary containing the info file
    info_file = create_info_file(
        sf_flavors=LHAPDF_ID,
        x_grids=grids_info_specs["x_grids"],
        q2_grids=grids_info_specs["q2_grids"],
        nrep=grids_info_specs["nrep"],
    )

    all_blocks = []
    xgrid = grids_info_specs["x_grids"]
    for pred in track(all_replicas, description="Looping over Replicas:"):
        all_singular_blocks = []
        block = generate_block(
            lambda pid, x, q2, pred=pred: pred[q2][pid][xgrid.index(x)],
            xgrid=grids_info_specs["x_grids"],
            Q2grid=grids_info_specs["q2_grids"],
            pids=LHAPDF_ID,
        )
        all_singular_blocks.append(block)
        all_blocks.append(all_singular_blocks)

    dump_set(name, info_file, all_blocks)


def main(
    model: pathlib.Path,
    a_value_spec: Union[None, int],
    x_dic_specs: Union[None, dict],
    q2_dic_specs: Union[None, dict],
    output: str,
    install_lhapdf: bool = True,
):

    a_value = a_value_spec if a_value_spec is not None else A_VALUE
    x_grids = x_dic_specs if x_dic_specs is not None else X_GRIDS
    q2_grids = q2_dic_specs if q2_dic_specs is not None else Q2_GRIDS

    _logger.info("Computing the blocks of the interpolation grids.")
    grid_info, prediction_allreplicas = parse_nn_predictions(
        model=model,
        a_value_spec=a_value,
        x_specs=x_grids,
        q2_dic_specs=q2_grids,
    )
    _logger.info("Dumping the blocks into files.")
    dump_pred_lhapdf(output, prediction_allreplicas, grid_info)
    if install_lhapdf:
        install_pdf(output)
        _logger.info("✓ The set has been successfully copied into LHAPDF.")
