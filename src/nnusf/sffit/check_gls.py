# -*- coding: utf-8 -*-
import numpy as np
from rich.progress import track

from .load_fit_data import get_predictions_q


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


def xf3_predictions(model_path, xgrid, q2_values, a_value):
    predictions_info = get_predictions_q(
        fit=model_path,
        a_slice=a_value,
        x_slice=xgrid.tolist(),
        qmin=q2_values.get("q2min", 1),
        qmax=q2_values.get("q2max", 5),
        n=q2_values.get("n", 1),
    )
    q2_grids = predictions_info.q
    predictions = predictions_info.predictions
    assert len(predictions) == xgrid.shape[0]
    assert isinstance(predictions, list)

    # Stack the list of x-values into a single np.array
    # The following returns as shape (nrep, nx, n, nsfs)
    predictions = [p[:, :, 2] for p in predictions]
    stacked_pred = np.stack(predictions).swapaxes(0, 1)

    return q2_grids, stacked_pred


def compute_integral(xgrid, weights_array, q2grids, xf3_nu):
    nb_q2points = q2grids.shape[0]
    xf3nu_perq2 = np.split(xf3_nu, nb_q2points, axis=1)
    results = []
    for xf3pred in xf3nu_perq2:
        divide_x = xf3pred.squeeze() / xgrid
        results.append(np.sum(divide_x * weights_array))
    return np.array(results)


def compute_gls_constant(nf_value, q2_value, n_loop=2):
    """The definitions below are taken from the following
    paper https://arxiv.org/pdf/hep-ph/9405254.pdf
    """
    lambda_msbar = 0.340  # in GeV

    def a_nf(nf_value):
        return 4.583 - 0.333 * nf_value

    def b_nf(nf_value):
        return 41.441 - 8.020 * nf_value + 0.177 * pow(nf_value, 2)

    def alphas(nf_value, q2_value, n_loop):
        beta_zero = 11 - (2 * nf_value) / 3
        ratio_logscale = np.log(q2_value / lambda_msbar**2)
        prefac = 4 * np.pi / (beta_zero * ratio_logscale)

        mode_alphas = 0
        if n_loop >= 1:
            mode_alphas += 1
        if n_loop >= 2:
            beta_one = 102 - (38 * nf_value) / 3
            num = beta_one * np.log(ratio_logscale)
            den = pow(beta_zero, 2) * ratio_logscale
            mode_alphas += num / den
        if n_loop >= 3:
            raise ValueError("Order not accounted yet!")

        return prefac * mode_alphas

    norm_alphas = alphas(nf_value, q2_value, n_loop) / np.pi
    return 3 * (
        1
        - norm_alphas
        - a_nf(nf_value) * pow(norm_alphas, 2)
        - b_nf(nf_value) * pow(norm_alphas, 3)
    )


def check_gls_sumrules(fit, nx, q2_values_dic, a_value, *args, **kwargs):
    del args
    del kwargs

    xgrid, weights = gen_integration_input(nx)
    q2grids, xf3nu = xf3_predictions(fit, xgrid, q2_values_dic, a_value)

    xf3nu_int = []
    for r in track(xf3nu, description="Looping over Replicas:"):
        xf3nu_int.append(compute_integral(xgrid, weights, q2grids, r))
    gls_results = compute_gls_constant(3, q2grids, n_loop=2)

    return q2grids, gls_results, np.stack(xf3nu_int)
