# -*- coding: utf-8 -*-
import math
import numpy as np
from eko.couplings import Couplings
from eko.io import types as ekotypes
from rich.progress import track

from .load_fit_data import get_predictions_q


def gen_integration_input(nx_specs):
    """Generate the points and weights for the integration."""
    nb_points = nx_specs.get("nx", 100)
    xmin_log = nx_specs.get("xmin_log", -2)
    lognx = int(nb_points / 3)
    linnx = int(nb_points - lognx)
    xgrid_log = np.logspace(xmin_log, -1, lognx + 1)
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

    # Compute the average of the xF3 predictions
    avg = [(p[:, :, 2] + p[:, :, 5]) / 2 for p in predictions]
    # Stack the list of x-values into a single np.array
    # The following returns as shape (nrep, nx, n, nsfs)
    stacked_pred = np.stack(avg).swapaxes(0, 1)

    return q2_grids, stacked_pred


def compute_integral(xgrid, weights_array, q2grids, xf3_avg):
    nb_q2points = q2grids.shape[0]
    xf3avg_perq2 = np.split(xf3_avg, nb_q2points, axis=1)
    results = []
    for xf3pred in xf3avg_perq2:
        divide_x = xf3pred.squeeze() / xgrid
        results.append(np.sum(divide_x * weights_array))
    return np.array(results)


def compute_gls_constant(nf_value, q2_value, n_loop=3):
    """The definitions below are taken from the following
    paper https://arxiv.org/pdf/hep-ph/9405254.pdf
    """

    def a_nf(nf_value):
        return 4.583 - 0.333 * nf_value

    def b_nf(nf_value):
        return 41.441 - 8.020 * nf_value + 0.177 * pow(nf_value, 2)

    def alphas_eko(q2_value, order=3):
        # set the (alpha_s, alpha_em) reference values
        alphas_ref = ekotypes.FloatRef(value=0.118, scale=91.0)
        alphaem_ref = ekotypes.FloatRef(value=0.007496252, scale=math.nan)
        couplings_ref = ekotypes.CouplingsRef(
            alphas=alphas_ref,
            alphaem=alphaem_ref,
            num_flavs_ref=None,
            max_num_flavs=5,
        )

        # set heavy quark masses and their threshold ratios
        heavy_quark_masses = np.power([1.51, 4.92, 172.0], 2)
        thresholds_ratios = [1.0, 1.0, 1.0]

        # set (QCD,QED) perturbative order
        order = (order, 1)

        strong_coupling = Couplings(
            couplings_ref,
            order,
            ekotypes.CouplingEvolutionMethod.EXACT,
            heavy_quark_masses,
            ekotypes.QuarkMassSchemes.POLE,
            thresholds_ratios,
        )

        results = [strong_coupling.a_s(q) for q in q2_value]

        return 4 * np.pi * np.asarray(list(results))

    norm_alphas = alphas_eko(q2_value, order=n_loop) / np.pi
    print(norm_alphas)
    return 3 * (
        1
        - norm_alphas
        - a_nf(nf_value) * pow(norm_alphas, 2)
        - b_nf(nf_value) * pow(norm_alphas, 3)
    )


def check_gls_sumrules(fit, nx_specs, q2_values_dic, a_value, *args, **kwargs):
    del args
    del kwargs

    xgrid, weights = gen_integration_input(nx_specs)
    q2grids, xf3avg = xf3_predictions(fit, xgrid, q2_values_dic, a_value)

    xf3avg_int = []
    for r in track(xf3avg, description="Looping over Replicas:"):
        xf3avg_int.append(compute_integral(xgrid, weights, q2grids, r))
    gls_results = compute_gls_constant(5, q2grids, n_loop=2)

    return q2grids, gls_results, np.stack(xf3avg_int)
