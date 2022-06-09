# -*- coding: utf-8 -*-
"""Filter data from original raw tables.

Data are then provided in a custom "CommonData" format (specific to this
structure function project).

"""

import importlib.util
import logging
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
        dataset_name = "_".join(dataset.stem.strip("DATA_").lower().split("_")[:-1])
        plugin_name = f"filter_{dataset_name}"

        spec = importlib.util.spec_from_file_location(
            plugin_name,
            (path_to_commondata / "filters" / plugin_name).with_suffix(".py"),
        )
        # We do not really want to fail at this point
        if spec is None:
            _logger.error(f"Filter for '{dataset}' not implemented yet!")
            continue

        plugin_module = importlib.util.module_from_spec(spec)
        __import__("pdb").set_trace()
        plugin_module.main(path_to_commondata)
