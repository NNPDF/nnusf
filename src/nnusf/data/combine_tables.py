import pandas as pd

from pathlib import Path
from nnusf.data.loader import Loader
from rich.console import Console


console = Console()

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
        console.print(f"\n• Experiment: {dataset}", style="bold red")
        for obs in MAP_DATASET_OBS[dataset]:
            console.print(f"[+] Observables: {obs}", style="bold blue")
            load_class = Loader(DATA_PATH, THEORY_PATH, dataset, obs)
            combined_tables.append(load_class.load())
    concatenated_tables = pd.concat(combined_tables)
    concatenated_tables.to_csv(DATA_PATH / "combined_tables.csv")


def main():
    dataset_lists = ["BEBCWA59", "CCFR", "CHARM", "NUTEV", "CHORUS", "CDHSW"]
    combine_tables(dataset_lists)
