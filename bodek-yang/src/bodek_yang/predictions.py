import logging
import pathlib
import tempfile
import tarfile

import lhapdf
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pineappl

from . import load, utils

logger = logging.getLogger(__name__)


def main(grids: pathlib.Path, pdf: str, err: str = "theory"):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir).absolute()

        # extract tar content
        if grids.suffix == ".tar":
            utils.extract_tar(grids, tmpdir)
            grids = tmpdir / "grids"

        preds_dest = tmpdir / "predictions"
        preds_dest.mkdir()

        genie = load.load()
        q2grid, xgrid = np.meshgrid(*load.kin_grids())
        gmask = load.mask()

        prescr = utils.nine_points
        xpoint = 10

        for gpath in grids.iterdir():
            if "pineappl" not in gpath.name:
                continue
            obs = gpath.stem.split(".")[0]
            kind = obs.split("_")[0]

            grid = pineappl.grid.Grid.read(gpath)

            plt.figure()

            if err == "theory":
                # theory uncertainties
                pdfset = lhapdf.mkPDF(pdf)
                pred = grid.convolute_with_one(
                    2212, pdfset.xfxQ2, pdfset.alphasQ2, xi=prescr
                )
                pred = pred.reshape((*xgrid.shape, len(prescr)))

                central = 4
                bulk = slice(0, -1)
                err_source = "9 pts."
            elif err == "pdf":
                # PDF uncertainties
                pred = []
                for pdfset in lhapdf.mkPDFs(pdf):
                    member_pred = grid.convolute_with_one(
                        2212, pdfset.xfxQ2, pdfset.alphasQ2
                    )
                    pred.append(member_pred)

                pred = np.array(pred).T.reshape((*xgrid.shape, len(pred)))

                central = 0
                bulk = slice(1, -1)
                err_source = "PDF replicas"
            else:
                raise ValueError(f"Invalid error type '{err}'")

            plt.plot(
                q2grid[xpoint],
                pred[xpoint, :, central],
                color="tab:blue",
                label="yadism",
            )
            plt.fill_between(
                q2grid[xpoint],
                pred[xpoint, :, bulk].min(axis=1),
                pred[xpoint, :, bulk].max(axis=1),
                facecolor=clr.to_rgba("tab:blue", alpha=0.1),
                label=err_source,
            )
            np.save(preds_dest / obs, pred)

            genie_pred = genie[f"{kind}_free_p"]
            genie_pred = genie_pred[gmask].reshape(xgrid.shape)
            plt.scatter(
                q2grid[xpoint],
                genie_pred[xpoint],
                color="tab:red",
                marker="x",
                label="genie",
            )

            name, qualifier = obs.split("_")
            plt.title(f"$F_{{{name[1]},{qualifier}}}(x = {xgrid[xpoint, 0]:.3g})$")
            plt.xscale("log")
            plt.legend()

            plt.savefig(preds_dest / f"{obs}.pdf")

        with tarfile.open(pathlib.Path.cwd() / "predictions.tar", "w") as tar:
            for path in tmpdir.iterdir():
                tar.add(path.absolute(), path.relative_to(tmpdir))
