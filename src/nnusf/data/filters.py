# -*- coding: utf-8 -*-
"""Filter data from original raw tables.

Data are then provided in a custom "CommonData" format (specific to this
structure function project).

"""

import importlib
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def main(list_of_datasets: list[Path]) -> None:
    """Filter all the datasets at once.

    It will filter all the requested datasets, starting from the raw tables
    provided by experimental collaborations, and dump the corresponding tables.

    Parameters
    ----------
    list_of_datasets: list[Path]
        list containing the path to all the datasets

    """
    for dataset in list_of_datasets:
        _logger.info(f"Filter dataset from the '{dataset}' experiment")

        path_to_commondata = dataset.parents[1]
        exp = "_".join(dataset.stem.strip("DATA_").lower().split("_")[:-1])
        mod_name = f"filter_{exp}"

        try:
            sys.path.insert(0, str((path_to_commondata / "filters").absolute()))
            mod = importlib.import_module(mod_name)
            mod.main(path_to_commondata)
        # We do not really want to fail at this point
        except ModuleNotFoundError:
            _logger.error(f"Filter for '{mod_name}' not implemented yet!")
        finally:
            sys.path.pop(0)
