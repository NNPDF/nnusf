import numpy as np
import numpy.typing as npt
from scipy import interpolate, integrate


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


def lhapdf(pdf, pid: int):
    xgrid = np.geomspace(pdf.xMin, pdf.xMax, 50)
    q2grid = np.geomspace(pdf.q2Min, pdf.q2Max, 50)
    xg2, qg2 = np.meshgrid(xgrid, q2grid)

    values = np.array(pdf.xfxQ2(pid, xg2.flatten(), qg2.flatten())).reshape(
        (xgrid.size, q2grid.size)
    )
    return xs(values, xgrid=xgrid, ygrid=q2grid)
