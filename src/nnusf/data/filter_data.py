# -*- coding: utf-8 -*-

import logging

from importlib import import_module
from pathlib import Path

_logger = logging.getLogger(__name__)


class MissingFilter(Exception):
    pass


def main(list_of_datasets: list[Path]) -> None:
    """Filter all the datasets at once and generate the corresponding
    Pandas tables.

    Parameters
    ----------
    list_of_datasets: list[Path]
        list containing the path to all the datasets
    """
    for dataset in list_of_datasets:
        _logger.info(f"Filter dataset from the {dataset} experiment")
        path_to_commondata = dataset.parents[1]
        dataset_name = dataset.stem.strip("DATA_").lower()
        plugin_name = f"nnusf.data.filters.filter_{dataset_name}"
        try:
            plugin_module = import_module(plugin_name)
            plugin_module.main(path_to_commondata)
        # We do not really want to fail at this point
        except ModuleNotFoundError:
            _logger.warning("Filter not implemented yet!")
