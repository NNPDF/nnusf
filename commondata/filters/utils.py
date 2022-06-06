import pandas as pd
from pathlib import Path


def write_to_csv(path: Path, exp_name:str, file: pd.DataFrame) -> None:
    file.to_csv(f"{path}/{exp_name}.csv", encoding="utf-8")


def construct_uncertainties(full_obs_errors: list, ERR_DESC: dict) -> pd.DataFrame:
    header_struct = pd.MultiIndex.from_tuples(
        [(k, v["treatment"], v["type"]) for k,v in ERR_DESC.items()],
        names=["name", "treatment", "type"]
    )
    full_error_values = pd.DataFrame(full_obs_errors).values
    errors_pandas_table = pd.DataFrame(
        full_error_values,
        columns=header_struct,
        index=range(1, len(full_obs_errors) + 1)
    )
    errors_pandas_table.index.name = "index"
    return errors_pandas_table


def dump_info_file(path : Path, exp_name : str, obs_list: list) -> None:
    info_folder = path.joinpath("info")
    info_folder.mkdir(exist_ok=True)
    df = pd.DataFrame(obs_list)
    df.to_csv(f"{info_folder}/{exp_name}.csv", encoding="utf-8")

