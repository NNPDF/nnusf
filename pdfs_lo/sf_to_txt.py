# -*- coding: utf-8 -*-
import pathlib
import shutil
import sys
import tarfile
import tempfile

import numpy as np


def broadgrid(ar, shape):
    return np.broadcast_to(ar[:, :, np.newaxis], shape)


def replica(shape):
    return np.broadcast_to(
        np.arange(shape[2])[np.newaxis, np.newaxis, :], shape
    )


tarpath = pathlib.Path(sys.argv[1])
files = {
    "F2": "F2_total.npy",
    "F3": "F3_total.npy",
    "x": "xgrid.npy",
    "q2": "q2grid.npy",
}

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = pathlib.Path(tmpdir)

    with tarfile.open(tarpath) as tar:
        tar.extractall(tmpdir)

    loaded = {k: np.load(tmpdir / "predictions" / v) for k, v in files.items()}

    txtfiles = []
    for lab, f in filter(lambda el: el[0][0] == "F", loaded.items()):
        stacked = (
            np.transpose(
                np.stack(
                    (
                        replica(f.shape),
                        broadgrid(loaded["x"], f.shape),
                        broadgrid(loaded["q2"], f.shape),
                        f,
                    )
                ),
                (0, 3, 2, 1),
            )
            .reshape((4, -1))
            .T
        )
        txtfiles.append(tmpdir / f"{lab}.txt")
        np.savetxt(
            txtfiles[-1],
            stacked,
            header=f"replica x Q2 {lab}",
            fmt="%d %e %e %e",
        )

    with tarfile.open(tarpath, "a") as tar:
        for path in txtfiles:
            tar.add(path, f"predictions/{path.name}")

    shutil.rmtree(tmpdir)
