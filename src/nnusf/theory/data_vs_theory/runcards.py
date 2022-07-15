# -*- coding: utf-8 -*-
"""Generate runcards for large yadism predictions.
"""
import logging
import pathlib
from typing import Optional

from ..highq import runcards

_logger = logging.getLogger(__name__)


def observables(datasets: list[str], path: Optional[pathlib.Path]) -> dict:
    """Collect all yadism runcards.

    Returns
    -------
    dict
        id to observables runcard mapping

    """
    if len(datasets) == 0 or path is None:
        if path is None:
            _logger.warning("No path to data folder provided.")
        else:
            _logger.warning("No data requested.")

    return runcards.overlap(datasets, path)
