# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd

ERR_DESC = {
    "stat": {
        "treatment": "ADD",
        "type": "UNCORR",
        "description": "Total statistical uncertainty",
    },
    "syst": {
        "treatment": "ADD",
        "type": "CORR",
        "description": "Total systematic uncertainty",
    },
}


def write_to_csv(path: Path, exp_name: str, file: pd.DataFrame) -> None:
    file.to_csv(f"{path}/{exp_name}.csv", encoding="utf-8")


def construct_uncertainties(full_obs_errors: list) -> pd.DataFrame:
    header_struct = pd.MultiIndex.from_tuples(
        [(k, v["treatment"], v["type"]) for k, v in ERR_DESC.items()],
        names=["name", "treatment", "type"],
    )
    full_error_values = pd.DataFrame(full_obs_errors).values
    errors_pandas_table = pd.DataFrame(
        full_error_values,
        columns=header_struct,
        index=range(1, len(full_obs_errors) + 1),
    )
    return errors_pandas_table


def build_obs_dict(fx: str, table: list, pid: float) -> dict:
    return {"type": fx, "tables": table, "projectile": pid}


def dump_info_file(path: Path, exp_name: str, obs_list: list, target=None) -> None:
    info_folder = path.joinpath("info")
    info_folder.mkdir(exist_ok=True)
    df = pd.DataFrame(obs_list)
    df["target"] = target
    df.to_csv(f"{info_folder}/{exp_name}.csv", encoding="utf-8")
