"""Script to produce the Yadism appendix benchmark plot."""
import pathlib

import lhapdf
import numpy as np
import pandas as pd
import yadism
import yaml
from eko.interpolation import make_grid
from yadmark.benchmark.external import apfel_utils
from yadmark.data import observables

here = pathlib.Path(__file__).absolute().parent

obs_names = ["F2", "XSCHORUSCC"]
x_grid = make_grid(30,30, x_min=1e-4)
q2_grid = np.geomspace(6**2, 300**2, 60)

x_fixed = 0.01
q2_fixed = 100**2


def load_theory():
    with open(
        here / "../src/nnusf/theory/assets/theory_nnusf.yaml",
        "r",
    ) as file:
        th = yaml.safe_load(file)
    th["PTO"] = 2
    th["FNS"] = "FONLL-C"
    th["NfFF"] = 4
    return th


def load_observable(obs_names):
    obs = observables.default_card
    obs["interpolation_xgrid"] = x_grid.tolist()
    obs["prDIS"] = "CC"
    obs["ProjectileDIS"] = "neutrino"
    obs["TargetDIS"] = "isoscalar"
    obs["observables"] = {}
    kinematics = [{"x": float(x_fixed), "Q2": float(q2), "y": 0.5} for q2 in q2_grid]
    kinematics.extend(
        [{"x": float(x), "Q2": float(q2_fixed), "y": 0.5} for x in x_grid[4:-3]]
    )
    for fx in obs_names:
        obs["observables"][fx] = kinematics
    return obs


def run_yadism(theory, observables, pdf):
    output = yadism.run_yadism(theory, observables)
    yad_pred = output.apply_pdf(pdf)
    return yad_pred


def run_apfel(theory, observables, pdf):
    log = apfel_utils.compute_apfel_data(theory, observables, pdf)
    return log


def run_bench(obs_names):
    theory = load_theory()
    observables = load_observable(obs_names)
    pdf = lhapdf.mkPDF("NNPDF40_nnlo_as_01180")
    yad_log = run_yadism(theory, observables, pdf)
    apfel_log = run_apfel(theory, observables, pdf)

    for obs in obs_names:
        yad_df = pd.DataFrame(yad_log[obs]).rename(columns={"result": "yadism"})
        apfel_df = pd.DataFrame(apfel_log[obs]).rename(columns={"result": "apfel"})
        benc_df = pd.concat([yad_df, apfel_df], axis=1).T.drop_duplicates().T
        benc_df.to_csv(here / f"{obs}.csv")


if __name__ == "__main__":

    run_bench(obs_names)
