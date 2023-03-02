# -*- coding: utf-8 -*-
"""
Module that computes the Structure Function predictions from the NN saved
model and dump them as a LHAPDF-like grid.
"""

import logging
import pathlib
from typing import Optional, Union

import numpy as np
from rich.progress import track

from ..sffit.load_fit_data import get_predictions_q
from ..utils import ROUNDING, compute_lhapdf
from .utils import (
    create_info_file,
    dump_set,
    generate_block,
    install_pdf,
    make_lambert_grid,
)

_logger = logging.getLogger(__name__)

A_VALUE = 1
LHAPDF_ID = [1001, 1002, 1003, 2001, 2002, 2003, 3001, 3002, 3003]


def combine_medium_high_q2(low_q2pred: np.ndarray, high_q2pred: np.ndarray):
    # TODO: Find a clean way to sort predictions per x per replica
    _logger.info("Combining the medium and high-Q2 predictions...")
    # We need to make sure that the number of replica are the same
    if low_q2pred.shape[0] < high_q2pred.shape[0]:
        high_q2pred = high_q2pred[: low_q2pred.shape[0]]
    else:
        low_q2pred = low_q2pred[: high_q2pred.shape[0]]
    assert low_q2pred.shape[0] == high_q2pred.shape[0]

    # Return a concatenated array along the Q2 direction
    return np.concatenate([low_q2pred, high_q2pred], axis=1)


def parse_nn_predictions(
    model: pathlib.Path,
    a_value_spec: int,
    x_specs: dict,
    q2_dic_specs: dict,
    pdfname: Union[str, None],
    min_highq2: Optional[float] = None,
):
    x_grids = make_lambert_grid(
        x_min=x_specs["min"],
        x_max=x_specs["max"],
        n_pts=x_specs["num"],
    )
    prediction_info = get_predictions_q(
        fit=model,
        a_slice=a_value_spec,
        x_slice=x_grids.tolist(),
        qmin=q2_dic_specs["min"],
        qmax=q2_dic_specs["max"],
        n=q2_dic_specs["num"],
        q_spacing="geomspace",
    )
    predictions = np.asarray(prediction_info.predictions)
    # The predictions above is of shape (nx, nrep, n_q2, n_sfs)
    # and the moveaxis transforms it into (nrep, n_q2, n_x, n_sfs)
    predictions = np.moveaxis(predictions, [0, 1, 2], [2, 0, 1])

    # Compute central and preprend results to array -> (nrep+1, n_q2, n_x, n_sfs)
    central = np.expand_dims(np.mean(predictions, axis=0), axis=0)
    predictions = np.concatenate([central, predictions], axis=0)

    q2min = min_highq2 if min_highq2 is not None else 900  # in GeV
    # Check that the Q2min of the Yadism prediction is greater that Q2max
    # of the NNUSF predictions.
    assert prediction_info.q[-1] < q2min
    q2grid = np.geomspace(q2min, 1e5, num=50).tolist()

    # Append the average to the array block
    copied_pred = np.copy(predictions)
    for i in range(copied_pred.shape[-1] // 2):
        avg = (copied_pred[:, :, :, i] + copied_pred[:, :, :, i + 3]) / 2
        average = np.expand_dims(avg, axis=-1)
        predictions = np.concatenate([predictions, average], axis=-1)

    prediction_infoq2 = prediction_info.q.tolist()
    if pdfname is not None:
        prediction_infoq2 += q2grid
        # Construct the Yadism predictions at high-Q2
        highq2_pred, _ = compute_lhapdf(
            pdfname,
            prediction_info.x,
            q2grid,
            LHAPDF_ID,
        )
        predictions = combine_medium_high_q2(predictions, highq2_pred)

    # Concatenate the Q2 values to dump into the LHAPDF grid
    prediction_infoq2 = [round(q, ROUNDING) for q in prediction_infoq2]
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
        "x_grids": prediction_info.x,
        "q2_grids": prediction_infoq2,
        "nrep": len(combined_replica),
    }

    return grids_info_specs, combined_replica


def dump_pred_lhapdf(
    name: str,
    am: int,
    all_replicas: list,
    grids_info_specs: dict,
    lhapids: list,
):
    # Generate the dictionary containing the info file
    info_file = create_info_file(
        sf_flavors=lhapids,
        a_value=int(am),
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
            pids=lhapids,
        )
        all_singular_blocks.append(block)
        all_blocks.append(all_singular_blocks)

    dump_set(name, info_file, all_blocks)


def main(
    model: pathlib.Path,
    pdfname: str,
    a_value_spec: Union[None, int],
    x_dic_specs: dict,
    q2_dic_specs: dict,
    output: str,
    min_highq2: Optional[float] = None,
    install_lhapdf: bool = True,
):

    a_value = a_value_spec if a_value_spec is not None else A_VALUE
    _logger.info("Computing the blocks of the interpolation grids.")
    grid_info, prediction_allreplicas = parse_nn_predictions(
        model=model,
        a_value_spec=a_value,
        x_specs=x_dic_specs,
        q2_dic_specs=q2_dic_specs,
        pdfname=pdfname,
        min_highq2=min_highq2,
    )
    _logger.info("Dumping the blocks into files.")
    dump_pred_lhapdf(
        output, a_value, prediction_allreplicas, grid_info, LHAPDF_ID
    )
    if install_lhapdf:
        install_pdf(output)
        _logger.info("✓ The set has been successfully copied into LHAPDF.")
