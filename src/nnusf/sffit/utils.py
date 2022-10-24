# -*- coding: utf-8 -*-
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Union

import numpy as np
import pygit2
import tensorflow as tf
from rich.console import Console
from rich.style import Style
from rich.table import Table

console = Console()


ADAPTIVE_LR = [
    {"range": [100, 250], "lr": 0.025},
    {"range": [50, 100], "lr": 0.01},
    {"range": [40, 50], "lr": 0.0075},
    {"range": [40, 50], "lr": 0.005},
    {"range": [10, 30], "lr": 0.0025},
    {"range": [5, 10], "lr": 0.0015},
    {"range": [1, 5], "lr": 0.001},
]


@dataclass
class TrainingStatusInfo:
    """Class for storing info to be shared among callbacks
    (in particular prevents evaluating multiple times for each individual
    callback).
    """

    tr_dpts: dict
    vl_dpts: dict
    best_chi2: Union[float, None] = None
    vl_chi2: Union[float, None] = None
    chix: Union[list, None] = None
    chi2_history: Union[dict, None] = None
    loss_value: float = 1e5
    vl_loss_value: Union[float, None] = None
    best_epoch: Union[int, None] = None

    def __post_init__(self):
        self.tot_vl = sum(self.vl_dpts.values())
        self.nbdpts = sum(self.tr_dpts.values())


def add_git_info(runcard_dict: dict):
    """Add git info to the runcard."""
    repo = pygit2.Repository(pathlib.Path().cwd())
    commit = repo[repo.head.target]
    runcard_dict["git_info"] = str(commit.id)


def set_global_seeds(global_seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(global_seed)
    random.seed(global_seed)
    tf.random.set_seed(global_seed)
    np.random.seed(global_seed)


def modify_lr(tr_loss_val, lr):
    for dic in ADAPTIVE_LR:
        range, lrval = dic["range"], dic["lr"]
        check = range[0] <= tr_loss_val < range[1]
        if check and (lr > lrval):
            return lrval
    return lr


def mask_expdata(y, tr_mask, vl_mask):
    """_summary_

    Parameters
    ----------
    y : _type_
        _description_
    tr_mask : _type_
        _description_
    vl_mask : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return y[tr_mask], y[vl_mask]


def mask_coeffs(coeff, tr_mask, vl_mask):
    """_summary_

    Parameters
    ----------
    coeff : _type_
        _description_
    tr_mask : _type_
        _description_
    vl_mask : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return coeff[tr_mask], coeff[vl_mask]


def mask_covmat(covmat, tr_mask, vl_mask):
    """_summary_

    Parameters
    ----------
    covmat : _type_
        _description_
    tr_mask : _type_
        _description_
    vl_mask : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    trmat = covmat[tr_mask].T[tr_mask]
    vlmat = covmat[vl_mask].T[vl_mask]
    return trmat, vlmat


def chi2(invcovmat):
    """_summary_

    Parameters
    ----------
    covmat : _type_
        _description_
    nb_datapoints : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    incovmatf = tf.keras.backend.constant(invcovmat)

    def chi2_loss(exp_data, fit_pred):
        diff_prediction = exp_data - fit_pred
        right_dot = tf.tensordot(
            incovmatf, tf.transpose(diff_prediction), axes=1
        )
        result = tf.tensordot(diff_prediction, right_dot, axes=1)
        return result

    return chi2_loss


def chi2_logs(train_info, vl_loss, tr_dpts, vl_dpts, epoch, lr):
    tot_trpts = sum(tr_dpts.values())
    tot_vlpts = sum(vl_dpts.values())
    style = Style(color="white")
    title = f"Epoch {epoch:08d}: LR={lr:6.4f}"
    table = Table(
        show_header=True,
        header_style="bold green",
        title=title,
        style=style,
        title_style="bold cyan",
    )
    vl_loss = vl_loss if isinstance(vl_loss, list) else [vl_loss]
    table.add_column("Dataset", justify="left", width=30)
    table.add_column("ndat(tr)", justify="right", width=12)
    table.add_column("chi2(tr)/Ntr", justify="right", width=12)
    table.add_column("ndat(vl)", justify="right", width=12)
    table.add_column("chi2(vl)/Nvl", justify="right", width=12)
    tot_val = vl_loss[0] / tot_vlpts

    vl_datpts = []
    for key in train_info:
        if key == "loss":
            continue
        vl_datpts.append(vl_dpts[key.strip("_loss")])
    if len(vl_loss) == len(vl_datpts):
        vl_loss.insert(0, 1.0)
    sigma_val = (np.array(vl_loss[1:]) / vl_datpts).std()

    for idx, (key, value) in enumerate(train_info.items()):
        if key == "loss":
            continue

        dataset_name = key.strip("_loss")
        chi2_tr = value / tr_dpts[dataset_name]
        chi2_vl = vl_loss[idx] / vl_dpts[dataset_name]
        highlight = ""
        endhl = ""
        if chi2_vl > tot_val + sigma_val:
            endhl = "[/]"
            if chi2_vl > tot_val + 2 * sigma_val:
                highlight = "[red]"
            else:
                highlight = "[yellow]"
        table.add_row(
            f"{highlight}{dataset_name}{endhl}",
            f"{tr_dpts[dataset_name]}",
            f"{chi2_tr:.4f}",
            f"{vl_dpts[dataset_name]}",
            f"{highlight}{chi2_vl:.4f}{endhl}",
        )

    table.add_row(
        "Tot chi2",
        f"{sum(i for i in tr_dpts.values())}",
        f"{train_info['loss'] / tot_trpts:.4f}",
        f"{sum(i for i in vl_dpts.values())}",
        f"{tot_val:.4f}",
        style="bold magenta",
    )
    return table
