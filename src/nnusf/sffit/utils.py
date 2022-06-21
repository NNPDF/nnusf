import os
import random
import numpy as np
import tensorflow as tf

from rich.table import Table
from rich.style import Style
from rich.console import Console

console = Console()


def set_global_seeds(global_seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(global_seed)
    random.seed(global_seed)
    tf.random.set_seed(global_seed)
    np.random.seed(global_seed)


def generate_mask(ndata, frac=0.75):
    """_summary_

    Parameters
    ----------
    ndata : _type_
        _description_
    frac : float, optional
        _description_, by default 0.75

    Returns
    -------
    _type_
        _description_
    """
    trmax = int(frac * ndata)
    mask = np.concatenate(
        [
            np.ones(trmax, dtype=np.bool_),
            np.zeros(ndata - trmax, dtype=np.bool_),
        ]
    )
    np.random.shuffle(mask)
    tr_mask = mask
    vl_mask = ~mask
    return tr_mask, vl_mask


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
        right_dot = tf.tensordot(incovmatf, tf.transpose(diff_prediction), axes=1)
        result = tf.tensordot(diff_prediction, right_dot, axes=1)
        return result

    return chi2_loss


def monitor_validation(vl_model, kins, exp_data):
    """_summary_

    Parameters
    ----------
    vl_model : _type_
        _description_
    kins : _type_
        _description_
    exp_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    loss = vl_model.evaluate(x=kins, y=exp_data, verbose=0)
    return [loss] if isinstance(loss, float) else loss


def chi2_logs(train_info, vl_loss, tr_dpts, vl_dpts, epoch, lr):
    tot_trpts = sum(tr_dpts.values())
    tot_vlpts = sum(vl_dpts.values())
    style = Style(color="white")
    title = f"Epoch {epoch:08d}: LR={lr:6.4f}"
    table = Table(
        show_header=True,
        header_style="bold white",
        title=title,
        style=style,
        title_style="bold cyan",
    )
    vl_loss = vl_loss if isinstance(vl_loss, list) else [vl_loss]
    table.add_column(" ", justify="left", width=15)
    table.add_column("chi2(tr)/Ntr", justify="right", width=12)
    table.add_column("chi2(vl)/Nvl", justify="right", width=12)
    for idx, (key, value) in enumerate(train_info.items()):
        if key != "loss":
            dataset_name = key.strip("_loss")
            chi2_tr = value / tr_dpts[dataset_name]
            chi2_vl = vl_loss[idx] / vl_dpts[dataset_name]
            table.add_row(
                f"{dataset_name}",
                f"{chi2_tr:.4f}",
                f"{chi2_vl:.4f}",
            )
    table.add_row(
        "Tot chi2",
        f"{train_info['loss'] / tot_trpts:.4f}",
        f"{vl_loss[0] / tot_vlpts:.4f}",
        style="bold white",
    )
    return table
