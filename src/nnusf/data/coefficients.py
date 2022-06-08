# -*- coding: utf-8 -*-
import pathlib

import numpy as np
from yadism.esf.exs import xs_coeffs

from ..theory import runcards
from . import loader


def main(datapaths: list[pathlib.Path], destination: pathlib.Path):
    destination.mkdir(parents=True, exist_ok=True)
    print("Coefficients destination:", destination)

    print("Saving coefficients:")
    for dataset in datapaths:
        name = dataset.stem.strip("DATA_")

        print("\tloaded", name)
        data = loader.Loader(dataset.parents[1], None, name)
        if data.n_data == 0:
            print("\t  NO data found")
            continue

        coeffs = np.zeros((data.n_data, 6))

        proj = data.fulltables["projectile"].values[0]

        if proj < 0:
            pos = [3]
        elif proj == 0:
            pos = [0, 3]
        elif proj > 0:
            pos = [0]
        else:
            raise ValueError
        pos = np.array(pos)

        if data.data_type in ["F2", "F3"]:
            pos += 0 if data.data_type == "F2" else 2
            coeffs[:, pos] = 1
        else:
            exp = name.split("_")[0]
            try:
                xs = loader.MAP_EXP_YADISM[exp]
            except KeyError:
                print(f"\t  NO available cross-section for '{exp}'")
                continue

            yadcoeffs = []
            for kin in data.kinematics:
                y = 0.1
                th = runcards.theory()
                yadcoeffs.append(
                    xs_coeffs(
                        xs,
                        y=y,
                        x=kin[0],
                        Q2=kin[1],
                        params=dict(
                            projectilePID=proj,
                            M2target=th["MP"] ** 2,
                            M2W=th["MW"] ** 2,
                            GF=th["GF"],
                        ),
                    )
                )

            yadcoeffs = np.array(yadcoeffs)
            idx = np.sum(np.meshgrid(np.arange(3), pos), axis=0).flatten()

            coeffs[:, idx] = np.transpose(
                yadcoeffs[:, :, np.newaxis]
                * np.ones_like(pos)[np.newaxis, np.newaxis, :],
                [0, 2, 1],
            ).reshape(yadcoeffs.shape[0], yadcoeffs.shape[1] * pos.size)

        # if average, divide by factor
        coeffs[np.sign(proj) == 0] *= 1.0 / 2.0

        dest = (destination / name).with_suffix(".npy")
        np.save(dest, coeffs)
        print(f"\t  {coeffs.shape} saved in", dest.relative_to(pathlib.Path.cwd()))
