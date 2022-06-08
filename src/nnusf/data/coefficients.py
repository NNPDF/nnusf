# -*- coding: utf-8 -*-
import pathlib

import numpy as np

from . import loader


def main(datapaths: list[pathlib.Path], destination: pathlib.Path):
    destination.mkdir(parents=True, exist_ok=True)
    print("Coefficients destination:", destination)

    print("Saving coefficients:")
    for dataset in datapaths:
        name = dataset.stem.strip("DATA_")
        obs = name.split("_")[1]

        # TODO: parse everything
        if obs not in ["F2", "F3"]:
            print("\t skip", name)
            continue

        print("\tloaded", name)
        data = loader.Loader(dataset.parents[1], None, name)

        ndata = data.kinematics[0].shape[0]
        coeffs = np.zeros((ndata, 6))

        pos = 0 if obs == "F2" else 2
        coeffs[:, pos] = 1

        dest = (destination / name).with_suffix(".npy")
        np.save(dest, coeffs)
        print("\tsaved in", dest.relative_to(pathlib.Path.cwd()))
