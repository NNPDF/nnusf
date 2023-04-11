# -*- coding: utf-8 -*-
"""
Module that given a PDF set imposes isoscalarity.

TODO: Add exception for LHAPDF set that contains subgrids.
"""
import logging

import numpy as np

from ..lhapdf.dump_grids import dump_pred_lhapdf
from ..lhapdf.utils import install_pdf
from ..utils import ROUNDING, compute_lhapdf

_logger = logging.getLogger(__name__)


ISOSPIN_CONJUGATION = {-2: -1, -1: -2, 1: 2, 2: 1}


def impose_isoscalarity(proton: np.ndarray, pid: list) -> np.ndarray:
    """Compute the neutron bound predictions from the proton
    using isospin symmetry and construct the free isoscalar
    nucleon/nucleus.

    Parameters
    ----------
    proton: np.ndarray
        array of shape (nrep, n_q2, n_x, n_pid) containing the
        proton PDF
    pid: list
        lis of LHAPDF IDs

    Returns
    -------
    np.ndarray
        array of values for free isoscalar nucleon/nucleus
    """

    mapid = []
    for id in pid:
        if id in ISOSPIN_CONJUGATION.keys():
            newid = ISOSPIN_CONJUGATION[id]
        else:
            newid = id
        mapid.append(newid)

    # Construct the index from the swapping
    mapindex = [mapid.index(id) for id in pid]

    # construct the neutron bound PDF from the proton and
    # swap the flavours using the `mapindex` mask.
    neutron_bound = np.copy(proton).T
    neutron_bound = neutron_bound[mapindex].T

    return (proton + neutron_bound) / 2


def construct_block(predictions: np.ndarray, metadata: dict) -> list:
    """Construct the block of predictions to be dumped into LHAPDF
    grids. Each replica is represented in terms of stacked dictionary.

    Parameters
    ----------
    predictions: np.ndarray
        predictions of shape (nrep, n_q2, n_x, n_pids)
    metadata: dict
        dictionary containing the metadata for the grids

    Returns
    -------
    list
        list whose elements are stacked dictionaries
    """

    prediction_infoq2 = [round(q, ROUNDING) for q in metadata["q2_grids"]]

    combined_replica = []
    for replica in predictions:  # loop over the replica
        q2rep_dic = {}
        # Loop over the results for all Q2 values
        for idq, q2rep in enumerate(replica):
            sfs_q2rep = {}
            q2rep_idx = prediction_infoq2[idq]
            # loop over the different flavours
            for idx in range(q2rep.shape[-1]):
                sfs_q2rep[metadata["pid"][idx]] = q2rep[:, idx]
            q2rep_dic[round(q2rep_idx, ROUNDING)] = sfs_q2rep
        combined_replica.append(q2rep_dic)

    return combined_replica


def main(pdfname: str, a_value: int, install: bool) -> None:
    """Main function that dumps the isoscalarified LHAPDF set."""

    predictions, metadata = compute_lhapdf(pdfname)

    _logger.info("Imposing isoscalarity on the PDF set.")
    isopred = impose_isoscalarity(predictions, metadata["pid"])
    isoparsed = construct_block(isopred, metadata)

    outputname = f"{pdfname}_iso"

    _logger.info("Dumping the free isoscalar set as LHAPDF grid.")
    dump_pred_lhapdf(
        name=outputname,
        am=a_value,
        all_replicas=isoparsed,
        grids_info_specs=metadata,
        lhapids=metadata["pid"],
    )

    if install:
        install_pdf(outputname)
        _logger.info("✓ The set has been successfully copied into LHAPDF.")
