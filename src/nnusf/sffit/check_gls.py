# -*- coding: utf-8 -*-
import logging

import numpy as np

from .load_fit_data import get_predictions_q

_logger = logging.getLogger(__name__)


def gen_integration_input(nb_points):
    """Generate the points and weights for the integration."""
    lognx = int(nb_points / 3)
    linnx = int(nb_points - lognx)
    xgrid_log = np.logspace(-2, -1, lognx + 1)
    xgrid_lin = np.linspace(0.1, 1, linnx)
    xgrid = np.concatenate([xgrid_log[:-1], xgrid_lin])

    spacing = [0.0]
    for i in range(1, nb_points):
        spacing.append(np.abs(xgrid[i - 1] - xgrid[i]))
    spacing.append(0.0)

    weights = []
    for i in range(nb_points):
        weights.append((spacing[i] + spacing[i + 1]) / 2.0)
    weights_array = np.array(weights)

    return xgrid, weights_array


def xf3_predictions(model_path, xgrid, q2_value, a_value):
    predictions_info = get_predictions_q(
        fit=model_path,
        a_slice=a_value,
        x_slice=xgrid,
        qmin=q2_value,
        qmax=2 * q2_value,
        n=1,
    )
    n_sfs = predictions_info.n_sfs
    predictions = predictions_info.predictions

    lower_68 = np.sort(predictions, axis=0)[int(0.16 * n_sfs)]
    upper_68 = np.sort(predictions, axis=0)[int(0.84 * n_sfs)]
    mean_sfs = np.mean(predictions, axis=0)

    # assert n_sfs == 5
    return mean_sfs[:, 2], lower_68[:, 2], upper_68[:, 2]


def compute_integral(xgrid, weights_array, xf3_nu):
    # TODO: Split if there are more values of Q2
    divide_x = xf3_nu / xgrid
    return np.sum(divide_x * weights_array)


def compute_gls_constant(nf_value, q2_value):
    def a_nf(nf_values):
        return 1.0

    def b_nf(nf_value):
        return 1.0

    def alphas(nf_value, q2_value):
        return 1.121

    norm_alphas = alphas(nf_value, q2_value) / np.pi
    return 3 * (
        1
        - norm_alphas
        - a_nf(nf_value) * pow(norm_alphas, 2)
        - b_nf(nf_value) * pow(norm_alphas, 3)
    )


def main(model_path, nx, q2_values_dic, a_value):
    pass
