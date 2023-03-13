# -*- coding: utf-8 -*-
import lhapdf
import logging
import math
import sys
import numpy as np
from eko.couplings import Couplings
from eko.io import types as ekotypes
from rich.progress import track

from .load_fit_data import get_predictions_q


_logger = logging.getLogger(__name__)

M_PROTON = 0.938  # GeV
NC, TF = 3, 0.5
CF = (pow(NC, 2) - 1) / (2 * NC)


def gen_integration_input(nx_specs):
    """Generate the points and weights for the integration."""
    nb_points = nx_specs.get("nx", 100)
    xmin_log = nx_specs.get("xmin_log", -2)
    lognx = int(nb_points / 3)
    linnx = int(nb_points - lognx)
    xgrid_log = np.logspace(xmin_log, -1, lognx + 1)
    xgrid_lin = np.linspace(0.1, 1, linnx, endpoint=False)
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


def compute_integrand_sfs(pdfname, xgrid, q2dic, a_value, rule):
    """Compute the integrand by calling a PDF instead of using the
    NN models.
    """
    pdf_instances = lhapdf.mkPDFs(pdfname)

    q2grid = np.linspace(q2dic["q2min"], q2dic["q2max"], num=q2dic["n"])

    def _compute_gls(xgrid, q2grid, pdf_instances):
        # Construct the 2D grids to compute LHAPDF
        xg2, qg2 = np.meshgrid(xgrid, q2grid)
        comb_pdfs = []
        for pdf in pdf_instances:
            xf3 = pdf.xfxQ2(3003, xg2.flatten(), qg2.flatten())
            comb_pdfs.append(np.asarray(xf3).reshape((xgrid.size, q2grid.size)))
        return comb_pdfs

    def _compute_bjorken(xgrid, q2grid, pdf_instances):
        # Construct the 2D grids to compute LHAPDF
        xg2, qg2 = np.meshgrid(xgrid, q2grid)
        comb_pdfs = []
        for pdf in pdf_instances:
            sf_comb = []
            for pid in [[1001, 1002], [2001, 2002]]:
                fn = pdf.xfxQ2(pid[0], xg2.flatten(), qg2.flatten())
                fb = pdf.xfxQ2(pid[1], xg2.flatten(), qg2.flatten())
                fn = np.asarray(fn).reshape((xgrid.size, q2grid.size))
                fb = np.asarray(fb).reshape((xgrid.size, q2grid.size))
                sf_comb.append(fn - fb)
            comb_pdfs.append(sf_comb[0] - sf_comb[1])
        return comb_pdfs

    comb_pdfs = locals()[f'_compute_{rule.lower()}'](xgrid, q2grid, pdf_instances)
    return q2grid, np.asarray(comb_pdfs)


def compute_integrand(model_path, rules, xgrid, q2_values, a_value):
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

    if rules == "GLS":
        # Compute the average of the xF3 predictions
        avg = [(p[:, :, 2] + p[:, :, 5]) / 2 for p in predictions]
    elif rules == "Bjorken":
        # Compute the F1 structure w/o Target Mass Corrections (TMC)
        # The factor of 1/(2x) is accounted when computing the integral
        avg = [
            (p[:, :, 0] - p[:, :, 1]) - (p[:, :, 3] - p[:, :, 4])
            for p in predictions
        ]
    elif rules == "GDH":
        # Compute the F1 structure functions w/ extra factor w/o TMC
        # See Eq. (4.5) of this paper https://arxiv.org/pdf/2303.00723
        # The factor of 1/(2x) is accounted when computing the integral
        avg = [
            8 * (p[:, :, 0] - p[:, :, 1]) / q2_grids
            for p in predictions
        ]
    else:
        raise ValueError("The sum rule is unknown!")
    # Stack the list of x-values into a single np.array
    # The following returns as shape (nrep, nx, nq2)
    stacked_pred = np.stack(avg).swapaxes(0, 1)

    return q2_grids, stacked_pred


def compute_integral(xgrid, weights_array, q2grids, integrand, norm=1):
    nb_q2points = q2grids.shape[0]
    integrand_perq2 = np.split(integrand, nb_q2points, axis=1)
    results = []
    for q2pred in integrand_perq2:
        # norm=1 for GLS and norm=2 for Bjorken
        divide_x = q2pred.squeeze() / (norm * xgrid)
        results.append(np.sum(divide_x * weights_array))
    return np.array(results)


def alphas_eko(q2_value, order=3):
    """Compute the value of `alpha_s` up to 3 Loop using the
    class defined in EKO.

    Parameters:
    -----------
    q2_value: list or np.ndarray
        values of Q2 for which a_s will be computed

    order: int
        Order of the perturbative computations

    Returns:
    --------
    np.ndarray:
        a_s values corresponding to the Q2 values
    """
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


def compute_gls_constant(nf_value, q2_value, n_loop=3):
    """The definitions below are taken from the following
    paper https://arxiv.org/pdf/hep-ph/9405254.pdf
    """

    def a_nf(nf_value):
        return 4.583 - 0.333 * nf_value

    def b_nf(nf_value):
        return 41.441 - 8.020 * nf_value + 0.177 * pow(nf_value, 2)

    norm_alphas = alphas_eko(q2_value, order=n_loop) / np.pi
    return 3 * (
        1
        - norm_alphas
        - a_nf(nf_value) * pow(norm_alphas, 2)
        - b_nf(nf_value) * pow(norm_alphas, 3)
    )


def compute_gdh_constant(nf_value, q2_value, n_loop=3):
    """Compute the GDH limit when Q2->0. The following implements
    Eq. (45) from the following paper https://arxiv.org/pdf/2303.00723.
    """
    del nf_value, n_loop

    magnetic_moment = 3.21  # Proton (neutron=3.66)
    limit = -magnetic_moment / pow(M_PROTON, 2)
    return np.repeat(limit, q2_value.size)


def compute_bjorken_constant(nf_value, q2_value, n_loop=3):
    """The definitions below are taken from the following paper
    https://www.sciencedirect.com/science/article/pii/S0550321316301602
    """

    epsilon = q2_value / pow(M_PROTON, 2)

    # Massless coefficients up to 4-Loop/Order(a_s^4)
    def z_nf_massless(nf_value):
        """Coefficients of the a_s^0 massless term in the expansion."""
        return 1.0

    def a_nf_massless(nf_value):
        """Coefficients of the a_s^1 massless term in the expansion."""
        return -2 / 3

    def b_nf_massless(nf_value):
        """Coefficients of the a_s^2 massless term in the expansion."""
        return -4.8889 + 0.29630 * nf_value

    def c_nf_massless(nf_value):
        """Coefficients of the a_s^3 massless term in the expansion."""
        return -43.414 + 5.2623 * nf_value - 0.15947 * nf_value**2

    def d_nf_massless(nf_value):
        """Coefficients of the a_s^4 massless term in the expansion."""
        return -457.0 + 83.09 * nf_value - 5.03 * nf_value**2 + 0.1 * nf_value**3

    # Massless coefficients up to 2-Loop/Order(a_s^2)
    def z_nf_massive(q2_value):
        """Coefficients of the a_s^0 massive term in the expansion."""
        return epsilon / (1 + epsilon)

    def a_nf_massive(q2_value):
        """Coefficients of the a_s^1 massive term in the expansion."""
        ln = np.log(1 + epsilon)
        a = 2 + epsilon - 2 * pow(epsilon, 2)
        b = epsilon + pow(epsilon, 2)
        c = 1 + epsilon - 3 * pow(epsilon, 2)
        d = pow(epsilon, 2) + pow(epsilon, 3)
        return CF / 4 * (a / b - 2 * ln * c / d)

    def b_nf_massive(q2_value):
        """Coefficients of the a_s^2 massive term in the expansion."""
        n1, n2 = np.sqrt(1 + 4 / epsilon) + 1, np.sqrt(1 + 4 / epsilon) - 1
        ln = np.log(n1 / n2) * np.sqrt(1 + 4 / epsilon)

        a1 = -344 / (21 * epsilon) - 268 / 105 - (4 * epsilon) / 105
        a2 = (2 * pow(epsilon, 2)) / 105
        a = ln * (a1 + a2)
        b = (8 / 3 - (2 * pow(epsilon, 2)) / 105) * np.log(epsilon)
        c = 856 / (21 * epsilon) + 2258 / 315 - (4 * epsilon) / 105
        return (CF * TF) / 16 * (-8 / pow(epsilon, 2) * pow(ln, 2) + a + b + c)

    norm_alphas = alphas_eko(q2_value, order=n_loop) / np.pi
    return (
        z_nf_massive(q2_value)
        + a_nf_massive(q2_value) * norm_alphas
        + b_nf_massive(q2_value) * pow(norm_alphas, 2)
    )


def check_sumrule(fit, rule, nx_specs, q2_specs, a_value, *args, **kwargs):
    del args

    assert rule in ["GLS", "Bjorken", "GDH"]  # Make sure value is in the list
    xgrid, weights = gen_integration_input(nx_specs)

    sup_rules = ["Bjorken", "GLS"]
    if kwargs.get("pdf", None) is not None and rule not in sup_rules:
        _logger.warning(
            f"Using SF sets with {rule} sum rules is not supported. "
            "If this is not intended, remove 'pdf' key in the card."
        )
        sys.exit()

    # Choose whether to compute the Integrand from NN model or SFs
    if kwargs.get("pdf", None) is not None:
        q2grids, pred = compute_integrand_sfs(
            pdfname=kwargs["pdf"],
            xgrid=xgrid,
            q2dic=q2_specs,
            a_value=a_value,
            rule=rule,
        )
    else:
        q2grids, pred = compute_integrand(
            model_path=fit,
            rules=rule,
            xgrid=xgrid,
            q2_values=q2_specs,
            a_value=a_value,
        )

    norm = 1 if rule == "GLS" else 2
    pred_int = []
    for r in track(pred, description="Looping over Replicas:"):
        pred_int.append(compute_integral(xgrid, weights, q2grids, r, norm))

    nf = 3 if rule == "Bjorken" else 5
    call_function = globals()[f"compute_{rule.lower()}_constant"]
    gls_results = call_function(nf, q2grids, n_loop=3)

    return q2grids, gls_results, np.stack(pred_int)


def effective_charge(fit, rule, nx_specs, q2_specs, a_value, *args, **kwargs):
    """Compute the Unpolarized effective charge."""

    # Compute the Integral depending on the sum rule
    q2grids, _, integral = check_sumrule(
        fit=fit,
        rule=rule,
        nx_specs=nx_specs,
        q2_specs=q2_specs,
        a_value=a_value,
        *args,
        **kwargs,
    )
    _logger.info("Finished computing the intergal.")

    if rule == "GLS":
        # Expression below already normalized by Pi
        effective_as = (1 - integral / 3)
    elif rule == "Bjorken":
        # Expression below already normalized by Pi
        # effective_as = 3 / 2 * (1 - integral)
        effective_as = (1 - (6 / 1.276) * integral)
    else:
        raise ValueError("Rule not supported for Effective Charge.")

    return q2grids, effective_as
