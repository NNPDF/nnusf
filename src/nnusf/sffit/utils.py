# -*- coding: utf-8 -*-
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.style import Style
from rich.table import Table

console = Console()
_logger = logging.getLogger(__name__)


@dataclass
class TrainingStatusInfo:
    """Class for storing info to be shared among callbacks. In particular
    it prevents evaluating multiple times for each individual callback.
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


def set_global_seeds(global_seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(global_seed)
    random.seed(global_seed)
    tf.random.set_seed(global_seed)
    np.random.seed(global_seed)


def add_dict_json(replica_dir, extra_dict):
    """Dump the normalized experimental chi^2 values into the `fitinfo.json`.

    Parameters:
    -----------
    replica_dir: pathlib.Path
        path to the directory in which the results of a given replica is stored
    extra_dict: dict
        extra-dictionary to add to the existing .json file
    """
    with open(f"{replica_dir}/fitinfo.json", "r+") as fstream:
        json_file = json.load(fstream)
        json_file.update(extra_dict)
        # Sets file's current position at offset.
        fstream.seek(0)
        json.dump(json_file, fstream, sort_keys=True, indent=4)


def small_x_exp(model, w_name="small_x_preprocessing"):
    """Extract the weights of a the Small-x Preprocessing layer.

    Parameters:
    -----------
    model: tf.model
        tensorflow model
    w_name: str
        name of the layer

    Returns:
    --------
    list:
        list containing the values of the weights
    """
    corresp_layers = []
    _logger.info("Extracting small-x exponents from the weights.")
    for layer in model.layers:
        if w_name in layer.name and len(layer.weights) != 0:
            corresp_layers.append(layer.weights)
    # Select the first one as the second one correspond to
    # the one presend in the TheoryConstraint
    return [float(i) for i in corresp_layers[0]]


def subset_q2points(kin_unique, scaling_target, q2points, kincuts):
    """
    In order to smoothen the interpolation when mapping Q2 into [0, 1]
    we remove some of the Q2 points in the grid. This is done herein by
    removing
    """
    assert kin_unique.shape[0] == len(scaling_target)

    q2data_max = kincuts.get("q2max", 1e5)
    sub_smallq2 = q2points.get("small_q2points", 500)
    sub_largeq2 = q2points.get("large_q2points", 1e5)
    nt_q2points = kin_unique.shape[0]

    # Separate the Q2 from the real data to the matching
    nb_q2real = (kin_unique <= q2data_max).sum()
    nb_q2math = nt_q2points - nb_q2real

    if sub_largeq2 < nb_q2math:
        stepm = (nb_q2real // sub_smallq2) + 1
        stepn = (nb_q2math // sub_largeq2) + 1
        kin_unique = np.concatenate(
            [
                kin_unique[0:nb_q2real:stepm],
                kin_unique[nb_q2real:nt_q2points:stepn],
            ],
            axis=0,
        )
        scaling_target = np.concatenate(
            [
                scaling_target[0:nb_q2real:stepm],
                scaling_target[nb_q2real:nt_q2points:stepn],
            ]
        )
    else:
        _logger.error("Inconsistent values for number of sub-Q2 points")
    assert kin_unique.shape[0] == len(scaling_target)
    return kin_unique, scaling_target


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


def chi2_logs(train_info, vl_loss, tr_dpts, vl_dpts, epoch):
    """Instance of rich table that updates live the summary of
    the training.

    Parameters:
    -----------
    train_info: dict
        dictionary containing information on the training
    vl_loss: dict
        dictionary containing information on the validatio losses
    tr_dpts: dict
        contains the number of datapoints for all the training set
    vl_dpts:
        contains the number of datapoints for the validation set
    epoch: float
        the number of epochs
    """
    tot_trpts = sum(tr_dpts.values())
    tot_vlpts = sum(vl_dpts.values())
    style = Style(color="white")
    title = f"Epoch {epoch:08d}"
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
