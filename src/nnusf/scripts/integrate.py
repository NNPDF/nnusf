# -*- coding: utf-8 -*-
import logging
from typing import Optional

import lhapdf
import numpy as np
import numpy.typing as npt
from scipy import integrate, interpolate

_logger = logging.getLogger(__name__)


def xs(diffxs: npt.NDArray, xgrid: npt.NDArray, ygrid: npt.NDArray):
    interp_func = interpolate.RectBivariateSpline(xgrid, ygrid, diffxs)

    integral = integrate.dblquad(
        interp_func,
        ygrid.min(),
        ygrid.max(),
        lambda y_: xgrid.min(),
        lambda y_: xgrid.max(),
    )

    return integral


def integrate_lhapdf(
    pdfset: str,
    pid: int,
    xgrid: Optional[npt.NDArray] = None,
    q2grid: Optional[npt.NDArray] = None,
):
    pdf = lhapdf.mkPDF(pdfset, 0)  # Account only for the Central Value
    if xgrid is None:
        xgrid = np.geomspace(pdf.xMin, pdf.xMax, 50)
    if q2grid is None:
        q2grid = np.geomspace(pdf.q2Min, pdf.q2Max, 50)

    xg2, qg2 = np.meshgrid(xgrid, q2grid)
    values = np.array(pdf.xfxQ2(pid, xg2.flatten(), qg2.flatten())).reshape(
        (xgrid.size, q2grid.size)
    )

    return xs(values, xgrid=xgrid, ygrid=q2grid)


def main(
    pdfset: str,
    pid: int,
    xgrid: Optional[npt.NDArray] = None,
    q2grid: Optional[npt.NDArray] = None,
):
    integrated = integrate_lhapdf(pdfset, pid, xgrid, q2grid)
    _logger.info(f"Value of the intergration: {integrated}")
