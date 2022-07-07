# -*- coding: utf-8 -*-
"""Generate structure functions slices plots."""
import logging
import pathlib
import shutil
import tarfile
import tempfile

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .. import utils

_logger = logging.getLogger(__file__)


def plot(path: pathlib.Path) -> matplotlib.figure.Figure:
    """Do the actual plotting."""
    # load arrays
    xgrid = np.load(path.parent / "xgrid.npy")
    q2grid = np.load(path.parent / "q2grid.npy")
    obsgrid = np.load(path)

    # reduce amount of slices
    step = len(q2grid[0]) // 5
    x = xgrid[:, ::step][0]
    q2 = q2grid[:, 0]
    obs = obsgrid[:, ::step]

    spread = (obs.max() - obs.min()) * 1.5

    obs += (np.arange(obs.shape[1]) * spread)[np.newaxis, :, np.newaxis]
    obs = np.transpose(obs, (1, 0, 2))
    obsmean = obs.mean(axis=2)
    obsstd = obs.std(axis=2)
    obsplus = obsmean + obsstd
    obsminus = obsmean - obsstd

    fig = plt.figure()
    xmin = q2.min()
    xmax = q2.max()

    for idx, (values, plus, minus, xval) in enumerate(
        zip(obsmean, obsplus, obsminus, x)
    ):
        clr = -np.log(xval) / 8
        plt.hlines(
            idx * spread,
            xmin,
            xmax,
            linestyles="--",
            linewidth=0.5,
            color="0.8",
        )
        plt.plot(
            q2,
            values,
            label=f"{xval:.2e}",
            color=(clr, clr / 2, 0.2),
            linewidth=1,
        )
        plt.fill_between(q2, minus, plus, alpha=0.2, color=(clr, clr / 2, 0.2))
        plt.text(
            q2[-2], values[-1] - spread / 4, rf"$x=${xval:3.2e}", fontsize=10
        )

    plt.title(path.stem)
    plt.xlabel("$Q^2$")
    plt.ylabel(rf"$\mathcal{{O}} + {spread:3.2f} \cdot n_x$")

    ymax = obs.max()
    ymin = obs.min()
    ysize = ymax - ymin
    plt.ylim(ymin - ysize / 10, ymax + ysize / 10)

    #  plt.show()

    return fig


def main(source: pathlib.Path, kind: str, destination: pathlib.Path):
    """Create sliced plots."""
    utils.mkdest(destination)

    observables = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        with tarfile.open(source) as tarsrc:
            for name in tarsrc.getnames():
                path = pathlib.Path(name)
                if path.suffix != ".npy":
                    continue

                if any(k in path.stem for k in kind):
                    tarsrc.extract(name, tmpdir)
                    observables.append((tmpdir / name).absolute())
                if "grid" in path.stem:
                    tarsrc.extract(name, tmpdir)

        for obs in observables:
            fig = plot(obs)

            obsdest = destination / f"{obs.stem}.png"
            fig.savefig(obsdest)
            _logger.info(
                f"Saved 'sliced' plot to '{obsdest.relative_to(pathlib.Path.cwd())}'"
            )

        shutil.rmtree(tmpdir)
