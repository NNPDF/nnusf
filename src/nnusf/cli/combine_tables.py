import pandas as pd

from pathlib import Path
from nnusf.loader import Loader

ABS_PATH = Path(__file__).parents[3]
DATA_PATH = ABS_PATH.joinpath("commondata")
THEORY_PATH = ABS_PATH.joinpath("theory")

MAP_DATASET_OBS = {
    "BEBCWA59": ["F2", "F3"],
    "CCFR": ["F2", "F3"],
    "CHARM": ["F2", "F3"],
    "NUTEV": ["F2", "F3", "DXDYNUU", "DXDYNUB"],
    "CHORUS": ["F2", "F3", "DXDYNUU", "DXDYNUB"],
    "CDHSW": ["F2", "F3", "DXDYNUU", "DXDYNUB"],
}


def combine_tables(dataset_names: list[str]) -> None:
    """Combined all the tables with extra information.

    Parameters
    ----------
    dataset_names: list[str]
        list containing the names of the datasets
    """
    combined_tables = []
    for dataset in dataset_names:
        for obs in MAP_DATASET_OBS[dataset]:
            print(dataset, obs)
            load_class = Loader(DATA_PATH, THEORY_PATH, dataset, obs)
            values = load_class.load()[0]
            import pdb; pdb.set_trace()
            # combined_tables.append(load_class.load()[0])
    concatenated_tables = pd.concat(combined_tables)
    concatenated_tables.to_csv(DATA_PATH / "combined_tables.csv")


def main():
    dataset_lists = ["BEBCWA59"]
    combine_tables(dataset_lists)
