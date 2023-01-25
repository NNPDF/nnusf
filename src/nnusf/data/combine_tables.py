# -*- coding: utf-8 -*-
import logging
import pathlib

import pandas as pd

from .loader import Loader

_logger = logging.getLogger(__name__)


def combine_tables(datasets: list[pathlib.Path]) -> pd.DataFrame:
    """Combined all the tables with extra information.

    Parameters
    ----------
    dataset_names: list[str]
        list containing the names of the datasets
    """
    combined_tables = []
    for dataset in datasets:
        _logger.info(f"Append experiment '{dataset}'")
        data = Loader(dataset.stem.strip("DATA_"), dataset.parents[1])
        tables = data.table.drop(["y"], axis=1, errors="ignore")
        combined_tables.append(tables)

    return pd.concat(combined_tables)


def main(dataset_list: list[pathlib.Path], destination: pathlib.Path):
    cat_tab = combine_tables(dataset_list)
    file = destination / "combined_tables.csv"
    cat_tab.to_csv(file)
    _logger.info(
        f"Combination generated, saved in '{file.relative_to(pathlib.Path.cwd())}'"
    )
