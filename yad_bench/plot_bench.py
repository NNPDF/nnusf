"""Script to produce the Yadism appendix benchmark plot."""
import itertools
import pathlib

import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_bench import here, obs_names, q2_fixed, x_fixed

obs_label = {"F2": "$F_2$", "F2_total": "$F_2$", "XSCHORUSCC": r"$\sigma_{\nu}$"}


def flatten(d):
    newd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for ik, iv in flatten(v).items():
                newd[f"{k}.{ik}"] = iv
        else:
            newd[k] = v
    return newd


def load_style():
    path = here / "style.yaml"
    style = flatten(yaml.safe_load(pathlib.Path(path).read_text()))
    capstyle = "lines.solid_capstyle"
    prop_cycle = "axes.prop_cycle"
    if capstyle in style:
        style[capstyle] = mpl._enums.CapStyle(style[capstyle])
    pcd = {k: v for k, v in style.items() if prop_cycle in k}
    if len(pcd) > 0:
        length = max((len(l) for l in pcd.values()))
        for k, v in pcd.items():
            del style[k]
            cyc = cycler.cycler(
                k.split(".")[-1], itertools.islice(itertools.cycle(v), length)
            )
            if prop_cycle not in style:
                style[prop_cycle] = cyc
            else:
                style[prop_cycle] += cyc

    return style


plt.style.use(load_style())


def plot_obs(obs):

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2)
    ax_x = plt.subplot(gs[0, 0])
    ax_q2 = plt.subplot(gs[0, 1])
    df = pd.read_csv(f"{obs}.csv")
    df["Q"] = np.sqrt(df.Q2)
    dfq2 = df[df.x == x_fixed]
    dfx = df[df.Q2 == q2_fixed]
    dfx = dfx[:-2]

    ax_x.scatter(dfx.x, dfx.yadism / dfx.apfel, marker="x", label="Apfel")
    ax_q2.scatter(dfq2.Q, dfq2.yadism / dfq2.apfel, marker="x", label="Apfel")

    ax_x.set_ylabel("Yadism / Apfel")
    ax_x.set_xlabel("$x$")
    ax_q2.set_xlabel("$Q$")
    ax_x.set_title(f"{obs_label[obs]}($x$), $Q={np.sqrt(q2_fixed):.1f}$ GeV")
    ax_q2.set_title(f"{obs_label[obs]}($Q$), $x={x_fixed:.2f}$")
    ax_x.set_xscale("log")
    ax_q2.set_xscale("log")

    ax_x.hlines(
        1.0,
        dfx.x.min(),
        dfx.x.max(),
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    ax_q2.hlines(
        1.0,
        dfq2.Q.min(),
        dfq2.Q.max(),
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    plt.savefig(f"{obs}_bench.pdf")
    plt.tight_layout()


if __name__ == "__main__":

    for obs in obs_names:
        plot_obs(obs)
