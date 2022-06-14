import numpy as np
import tensorflow as tf


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
    vl_mask = mask == False
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


def chi2(covmat, nb_datapoints):
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
    invcovmat = np.linalg.inv(covmat)
    incovmatf = tf.keras.backend.constant(invcovmat)

    def chi2_loss(exp_data, fit_pred):
        diff_prediction = exp_data - fit_pred
        right_dot = tf.tensordot(incovmatf, tf.transpose(diff_prediction), axes=1)
        result = tf.tensordot(diff_prediction, right_dot, axes=1)
        return result / nb_datapoints

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
    return loss
