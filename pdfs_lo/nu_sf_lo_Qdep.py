#!/usr/bin/env python
# -*- coding: utf-8 -*-


import lhapdf
import matplotlib.pyplot as py
import numpy as np
from matplotlib import gridspec, rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
from pylab import *

# ---------------------------------------------------------
# ---------------------------------------------------------
# General plot settings
nq = 200
qmin = 1.65
qmax = 10
# set x grid
Q = np.logspace(np.log10(qmin), np.log10(qmax), nq)

print(Q)

# Number of PDF sets to be compared
nset = 3
# ----------------------------------------------------------------
# modernPDFs
# pdfset=["CT18NNLO","NNPDF40_nnlo_as_01180","MSHT20nnlo_as118"]
# pdfsetlab=[r"${\rm CT18}$",r"${\rm NNPDF4.0}$",r"${\rm MSHT20}$"]
# error_option=["ct","mc_68cl","mmht"]
# filelabel="-modernpdfs_q5gev"
# oldPDFs
pdfset = ["JR14NLO08VF", "NNPDF40_nnlo_as_01180", "cteq61"]
pdfsetlab = [r"${\rm JR14}$", r"${\rm NNPDF4.0}$", r"${\rm CTEQ6.1}$"]
error_option = ["symmhessian", "mc_68cl", "ct"]
filelabel = "-oldpdfs_x0p794"

# Value of x
x = 0.794

# Reduce verbosity of LHAPDF
lhapdf.setVerbosity(0)
# max number of replicas
nrepmax = 100
# Number of replicas
nrep = np.zeros(nset, dtype=int)
# number of Structure Functions to be plotted
nsf = 4
# Set x grid

# run over PDF sets
for iset in range(nset):
    # Initialise PDF set
    p = lhapdf.getPDFSet(pdfset[iset])
    print(p.description)
    nrep[iset] = int(p.get_entry("NumMembers")) - 1
    print("nrep =", nrep[iset])
    # Arrays to store LHAPDF results
    if iset == 0:
        fit1 = np.zeros((nrep[iset], nsf, nq))
        fit1_cv = np.zeros((nsf, nq))
    if iset == 1:
        fit2 = np.zeros((nrep[iset], nsf, nq))
        fit2_cv = np.zeros((nsf, nq))
    if iset == 2:
        fit3 = np.zeros((nrep[iset], nsf, nq))
        fit3_cv = np.zeros((nsf, nq))

    # Run over replicas
    for i in range(1, nrep[iset] + 1):
        p = lhapdf.mkPDF(pdfset[iset], i)

        # Run over x arrat
        for k in range(nq):

            q = Q[k]

            # run over flavours
            for isf in range(nsf):

                # ----------------------------------------------------------------
                if isf == 0:  # F2_nu_p
                    if iset == 0:
                        fit1[i - 1][isf][k] = 2 * (
                            p.xfxQ(-2, x, q)
                            + p.xfxQ(1, x, q)
                            + p.xfxQ(3, x, q)
                            + p.xfxQ(-4, x, q)
                        )
                    if iset == 1:
                        fit2[i - 1][isf][k] = 2 * (
                            p.xfxQ(-2, x, q)
                            + p.xfxQ(1, x, q)
                            + p.xfxQ(3, x, q)
                            + p.xfxQ(-4, x, q)
                        )
                    if iset == 2:
                        fit3[i - 1][isf][k] = 2 * (
                            p.xfxQ(-2, x, q)
                            + p.xfxQ(1, x, q)
                            + p.xfxQ(3, x, q)
                            + p.xfxQ(-4, x, q)
                        )
                # ----------------------------------------------------------------
                # ----------------------------------------------------------------
                if isf == 1:  # F2_nubar_p
                    if iset == 0:
                        fit1[i - 1][isf][k] = 2 * (
                            p.xfxQ(+2, x, q)
                            + p.xfxQ(-1, x, q)
                            + p.xfxQ(-3, x, q)
                            + p.xfxQ(+4, x, q)
                        )
                    if iset == 1:
                        fit2[i - 1][isf][k] = 2 * (
                            p.xfxQ(+2, x, q)
                            + p.xfxQ(-1, x, q)
                            + p.xfxQ(-3, x, q)
                            + p.xfxQ(+4, x, q)
                        )
                    if iset == 2:
                        fit3[i - 1][isf][k] = 2 * (
                            p.xfxQ(+2, x, q)
                            + p.xfxQ(-1, x, q)
                            + p.xfxQ(-3, x, q)
                            + p.xfxQ(+4, x, q)
                        )
                # ----------------------------------------------------------------
                # ----------------------------------------------------------------
                if isf == 2:  # xF3_nu_p
                    if iset == 0:
                        fit1[i - 1][isf][k] = 2 * (
                            -p.xfxQ(-2, x, q)
                            + p.xfxQ(1, x, q)
                            + p.xfxQ(3, x, q)
                            - p.xfxQ(-4, x, q)
                        )
                    if iset == 1:
                        fit2[i - 1][isf][k] = 2 * (
                            -p.xfxQ(-2, x, q)
                            + p.xfxQ(1, x, q)
                            + p.xfxQ(3, x, q)
                            - p.xfxQ(-4, x, q)
                        )
                    if iset == 2:
                        fit3[i - 1][isf][k] = 2 * (
                            -p.xfxQ(-2, x, q)
                            + p.xfxQ(1, x, q)
                            + p.xfxQ(3, x, q)
                            - p.xfxQ(-4, x, q)
                        )
                # ----------------------------------------------------------------
                # ----------------------------------------------------------------
                if isf == 3:  # xF3_nubar_p
                    if iset == 0:
                        fit1[i - 1][isf][k] = 2 * (
                            p.xfxQ(+2, x, q)
                            - p.xfxQ(-1, x, q)
                            - p.xfxQ(-3, x, q)
                            + p.xfxQ(+4, x, q)
                        )
                    if iset == 1:
                        fit2[i - 1][isf][k] = 2 * (
                            p.xfxQ(+2, x, q)
                            - p.xfxQ(-1, x, q)
                            - p.xfxQ(-3, x, q)
                            + p.xfxQ(+4, x, q)
                        )
                    if iset == 2:
                        fit3[i - 1][isf][k] = 2 * (
                            p.xfxQ(+2, x, q)
                            - p.xfxQ(-1, x, q)
                            - p.xfxQ(-3, x, q)
                            + p.xfxQ(+4, x, q)
                        )
                # ----------------------------------------------------------------

    # Central values
    p = lhapdf.mkPDF(pdfset[iset], 0)
    for k in range(nq):
        q = Q[k]

        for isf in range(nsf):

            # ----------------------------------------------------------------
            if isf == 0:  # F2_nu_p
                if iset == 0:
                    fit1_cv[isf][k] = 2 * (
                        p.xfxQ(-2, x, q)
                        + p.xfxQ(1, x, q)
                        + p.xfxQ(3, x, q)
                        + p.xfxQ(-4, x, q)
                    )
                if iset == 1:
                    fit2_cv[isf][k] = 2 * (
                        p.xfxQ(-2, x, q)
                        + p.xfxQ(1, x, q)
                        + p.xfxQ(3, x, q)
                        + p.xfxQ(-4, x, q)
                    )
                if iset == 2:
                    fit3_cv[isf][k] = 2 * (
                        p.xfxQ(-2, x, q)
                        + p.xfxQ(1, x, q)
                        + p.xfxQ(3, x, q)
                        + p.xfxQ(-4, x, q)
                    )
            # ----------------------------------------------------------------
            # ----------------------------------------------------------------
            if isf == 1:  # F2_nubar_p
                if iset == 0:
                    fit1_cv[isf][k] = 2 * (
                        p.xfxQ(+2, x, q)
                        + p.xfxQ(-1, x, q)
                        + p.xfxQ(-3, x, q)
                        + p.xfxQ(+4, x, q)
                    )
                if iset == 1:
                    fit2_cv[isf][k] = 2 * (
                        p.xfxQ(+2, x, q)
                        + p.xfxQ(-1, x, q)
                        + p.xfxQ(-3, x, q)
                        + p.xfxQ(+4, x, q)
                    )
                if iset == 2:
                    fit3_cv[isf][k] = 2 * (
                        p.xfxQ(+2, x, q)
                        + p.xfxQ(-1, x, q)
                        + p.xfxQ(-3, x, q)
                        + p.xfxQ(+4, x, q)
                    )
            # ----------------------------------------------------------------
            # ----------------------------------------------------------------
            if isf == 2:  # xF3_nu_p
                if iset == 0:
                    fit1_cv[isf][k] = 2 * (
                        -p.xfxQ(-2, x, q)
                        + p.xfxQ(1, x, q)
                        + p.xfxQ(3, x, q)
                        - p.xfxQ(-4, x, q)
                    )
                if iset == 1:
                    fit2_cv[isf][k] = 2 * (
                        -p.xfxQ(-2, x, q)
                        + p.xfxQ(1, x, q)
                        + p.xfxQ(3, x, q)
                        - p.xfxQ(-4, x, q)
                    )
                if iset == 2:
                    fit3_cv[isf][k] = 2 * (
                        -p.xfxQ(-2, x, q)
                        + p.xfxQ(1, x, q)
                        + p.xfxQ(3, x, q)
                        - p.xfxQ(-4, x, q)
                    )
            # ----------------------------------------------------------------
            # ----------------------------------------------------------------
            if isf == 3:  # xF3_nubar_p
                if iset == 0:
                    fit1_cv[isf][k] = 2 * (
                        p.xfxQ(+2, x, q)
                        - p.xfxQ(-1, x, q)
                        - p.xfxQ(-3, x, q)
                        + p.xfxQ(+4, x, q)
                    )
                if iset == 1:
                    fit2_cv[isf][k] = 2 * (
                        p.xfxQ(+2, x, q)
                        - p.xfxQ(-1, x, q)
                        - p.xfxQ(-3, x, q)
                        + p.xfxQ(+4, x, q)
                    )
                if iset == 2:
                    fit3_cv[isf][k] = 2 * (
                        p.xfxQ(+2, x, q)
                        - p.xfxQ(-1, x, q)
                        - p.xfxQ(-3, x, q)
                        + p.xfxQ(+4, x, q)
                    )
            # ----------------------------------------------------------------

# ---------------------------------------------------------------------
# Compute central values and uncertainties
# ---------------------------------------------------------------------

for iset in range(nset):

    # MC PDF sets, 68% CL intervals
    if error_option[iset] == "mc_68cl":

        if iset == 0:
            p1_high = np.nanpercentile(fit1, 84, axis=0)
            p1_low = np.nanpercentile(fit1, 16, axis=0)
            p1_mid = (p1_high + p1_low) / 2.0
            p1_error = (p1_high - p1_low) / 2.0
        elif iset == 1:
            p2_high = np.nanpercentile(fit2, 84, axis=0)
            p2_low = np.nanpercentile(fit2, 16, axis=0)
            p2_mid = (p2_high + p2_low) / 2.0
            p2_error = (p2_high - p2_low) / 2.0
        elif iset == 2:
            p3_high = np.nanpercentile(fit3, 84, axis=0)
            p3_low = np.nanpercentile(fit3, 16, axis=0)
            p3_mid = (p3_high + p3_low) / 2.0
            p3_error = (p3_high - p3_low) / 2.0

    # MC PDF sets, 99% CL intervals (all replicas)
    elif error_option[iset] == "mc_99cl":

        if iset == 0:
            p1_high = np.nanpercentile(fit1, 99.5, axis=0)
            p1_low = np.nanpercentile(fit1, 0.5, axis=0)
            p1_mid = np.median(fit1, axis=0)
            p1_error = (p1_high - p1_low) / 2.0
        elif iset == 1:
            p2_high = np.nanpercentile(fit2, 99.5, axis=0)
            p2_low = np.nanpercentile(fit2, 0.5, axis=0)
            p2_mid = (p2_high + p2_low) / 2.0
            p2_mid = np.median(fit2, axis=0)
            p2_error = (p2_high - p2_low) / 2.0
        elif iset == 2:
            p3_high = np.nanpercentile(fit3, 99.5, axis=0)
            p3_low = np.nanpercentile(fit3, 0.5, axis=0)
            p3_mid = (p3_high + p3_low) / 2.0
            p3_error = (p3_high - p3_low) / 2.0
        else:
            print("invalid option")
            exit()

    # MC PDF sets, one-sigma and mean
    elif error_option[iset] == "mc_1sigma":

        if iset == 0:
            p1_high = np.mean(fit1, axis=0) + np.std(fit1, axis=0)
            p1_low = np.mean(fit1, axis=0) - np.std(fit1, axis=0)
            p1_mid = np.mean(fit1, axis=0)
            p1_error = np.std(fit1, axis=0)
        elif iset == 1:
            p2_high = np.mean(fit2, axis=0) + np.std(fit2, axis=0)
            p2_low = np.mean(fit2, axis=0) - np.std(fit2, axis=0)
            p2_mid = np.mean(fit2, axis=0)
            p2_error = np.std(fit2, axis=0)
        elif iset == 2:
            p3_high = np.mean(fit3, axis=0) + np.std(fit3, axis=0)
            p3_low = np.mean(fit3, axis=0) - np.std(fit3, axis=0)
            p3_mid = np.mean(fit3, axis=0)
            p3_error = np.std(fit3, axis=0)
        else:
            print("invalid option")
            exit()

    # CT: asymmetric Hessian with then normalisation to one-sigma
    elif error_option[iset] == "ct" or error_option[iset] == "mmht":

        if iset == 0:
            p1_mid = np.mean(fit1, axis=0)
            p1_error = np.std(fit1, axis=0)
            neig = int(nrep[iset] / 2)  # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p1_mid[isf][ix] = fit1_cv[isf][ix]
                    p1_error[isf][ix] = 0  # initialisation
                    for ieg in range(neig):
                        # print(2*ieg+1,nrep[iset])
                        p1_error[isf][ix] = (
                            p1_error[isf][ix]
                            + (
                                fit1[2 * ieg + 1][isf][ix]
                                - fit1[2 * ieg][isf][ix]
                            )
                            ** 2.0
                        )
                    p1_error[isf][ix] = math.sqrt(p1_error[isf][ix]) / 2
                    if error_option[iset] == "ct":
                        p1_error[isf][ix] = (
                            p1_error[isf][ix] / 1.642
                        )  # from 90% to 68% CL
            p1_high = p1_mid + p1_error
            p1_low = p1_mid - p1_error
        elif iset == 1:
            p2_mid = np.mean(fit2, axis=0)
            p2_error = np.std(fit2, axis=0)
            neig = int(nrep[iset] / 2)  # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p2_mid[isf][ix] = fit2_cv[isf][ix]
                    p2_error[isf][ix] = 0  # initialisation
                    for ieg in range(neig):
                        # print(2*ieg+1,nrep[iset])
                        p2_error[isf][ix] = (
                            p2_error[isf][ix]
                            + (
                                fit2[2 * ieg + 1][isf][ix]
                                - fit2[2 * ieg][isf][ix]
                            )
                            ** 2.0
                        )
                    p2_error[isf][ix] = math.sqrt(p2_error[isf][ix]) / 2
                    if error_option[iset] == "ct":
                        p2_error[isf][ix] = (
                            p2_error[isf][ix] / 1.642
                        )  # from 90% to 68% CL
            p2_high = p2_mid + p2_error
            p2_low = p2_mid - p2_error
        elif iset == 2:
            p3_mid = np.mean(fit3, axis=0)
            p3_error = np.std(fit3, axis=0)
            neig = int(nrep[iset] / 2)  # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p3_mid[isf][ix] = fit3_cv[isf][ix]
                    p3_error[isf][ix] = 0  # initialisation
                    for ieg in range(neig):
                        p3_error[isf][ix] = (
                            p3_error[isf][ix]
                            + (
                                fit3[2 * ieg + 1][isf][ix]
                                - fit3[2 * ieg][isf][ix]
                            )
                            ** 2.0
                        )
                    p3_error[isf][ix] = math.sqrt(p3_error[isf][ix]) / 2
                    if error_option[iset] == "ct":
                        p3_error[isf][ix] = (
                            p3_error[isf][ix] / 1.642
                        )  # from 90% to 68% CL
            p3_high = p3_mid + p3_error
            p3_low = p3_mid - p3_error
        else:
            print("invalid option")
            exit()

    # HERAPDF: symmetric Hessian
    elif error_option[iset] == "symmhessian":

        if iset == 0:
            p1_mid = np.mean(fit1, axis=0)
            p1_error = np.std(fit1, axis=0)
            neig = int(nrep[iset])  # Number of eigenvectors
            # Run over x points and flavour
            for ix in range(nq):
                for isf in range(nsf):
                    p1_mid[isf][ix] = fit1_cv[isf][ix]  # Central Hessian value
                    p1_error[isf][ix] = 0  # initialisation
                    for ieg in range(neig):
                        p1_error[isf][ix] = (
                            p1_error[isf][ix]
                            + (fit1[ieg][isf][ix] - p1_mid[isf][ix]) ** 2.0
                        )
                    p1_error[isf][ix] = math.sqrt(p1_error[isf][ix])
            p1_high = p1_mid + p1_error
            p1_low = p1_mid - p1_error

        else:
            print("invalid option")
            exit()

    else:
        print("Invalid error option = ", error_option[iset])
        exit()

# ----------------------------------------------------------------------

# *****************************************************************************
# *****************************************************************************

print("\n ****** Plotting absolute Structure Functions ******* \n")

ncols, nrows = 2, 4
py.figure(figsize=(ncols * 5, nrows * 3.5))
gs = gridspec.GridSpec(nrows, ncols)
rescolors = py.rcParams["axes.prop_cycle"].by_key()["color"]
yranges = [[0, 0.01], [0, 0.08], [-0.0, 0.01], [-0.0, 0.07]]
labelpdf = [
    r"$F_2^{\nu p}(x,Q)$",
    r"$F_2^{\bar{\nu} p}(x,Q)$",
    r"$xF_3^{\nu p}(x,Q)$",
    r"$xF_3^{\bar{\nu} p}(x,Q)$",
]

for isf in range(nsf):

    ax = py.subplot(gs[isf])
    p1 = ax.plot(Q, p1_mid[isf], ls="dashed")
    ax.fill_between(Q, p1_high[isf], p1_low[isf], color=rescolors[0], alpha=0.2)
    p2 = ax.fill(np.NaN, np.NaN, color=rescolors[0], alpha=0.2)
    p3 = ax.plot(Q, p2_mid[isf], ls="solid")
    ax.fill_between(Q, p2_high[isf], p2_low[isf], color=rescolors[1], alpha=0.2)
    p4 = ax.fill(np.NaN, np.NaN, color=rescolors[1], alpha=0.2)
    p5 = ax.plot(Q, p3_mid[isf], ls="dashdot")
    ax.fill_between(Q, p3_high[isf], p3_low[isf], color=rescolors[2], alpha=0.2)
    p6 = ax.fill(np.NaN, np.NaN, color=rescolors[2], alpha=0.2)
    ax.set_xscale("linear")
    ax.set_xlim(qmin, qmax)
    ax.tick_params(which="both", direction="in", labelsize=12, right=True)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=4)
    ax.set_ylabel(labelpdf[isf], fontsize=21)
    ax.set_ylim(yranges[isf][0], yranges[isf][1])
    ax.set_xlabel(r"$Q~({\rm GeV})$", fontsize=20)
    ax.text(0.05, 0.85, r"$x=0.794$", fontsize=16, transform=ax.transAxes)

    ax.legend(
        [(p1[0], p2[0]), (p3[0], p4[0]), (p5[0], p6[0])],
        [pdfsetlab[0], pdfsetlab[1], pdfsetlab[2]],
        frameon="True",
        loc=1,
        prop={"size": 16},
    )

print("\n ****** Plotting ratios Structure Functions ******* \n")

yranges = [[0.5, 1.5], [0.5, 1.5], [0.3, 1.7], [0.3, 1.7]]
labelpdf = [
    r"$F_2^{\nu p}(x,Q)$",
    r"$F_2^{\bar{\nu} p}(x,Q)$",
    r"$xF_3^{\nu p}(x,Q)$",
    r"$xF_3^{\bar{\nu} p}(x,Q)$",
]

for isf in range(nsf):

    norm = p1_mid[isf]

    ax = py.subplot(gs[4 + isf])
    p1 = ax.plot(Q, p1_mid[isf] / norm, ls="dashed")
    ax.fill_between(
        Q,
        p1_high[isf] / norm,
        p1_low[isf] / norm,
        color=rescolors[0],
        alpha=0.2,
    )
    p2 = ax.fill(np.NaN, np.NaN, color=rescolors[0], alpha=0.2)
    p3 = ax.plot(Q, p2_mid[isf] / norm, ls="solid")
    ax.fill_between(
        Q,
        p2_high[isf] / norm,
        p2_low[isf] / norm,
        color=rescolors[1],
        alpha=0.2,
    )
    p4 = ax.fill(np.NaN, np.NaN, color=rescolors[1], alpha=0.2)
    p5 = ax.plot(Q, p3_mid[isf] / norm, ls="dashdot")
    ax.fill_between(
        Q,
        p3_high[isf] / norm,
        p3_low[isf] / norm,
        color=rescolors[2],
        alpha=0.2,
    )
    p6 = ax.fill(np.NaN, np.NaN, color=rescolors[2], alpha=0.2)
    ax.set_xscale("linear")
    ax.set_xlim(qmin, qmax)
    ax.tick_params(which="both", direction="in", labelsize=12, right=True)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=4)
    ax.set_ylabel(labelpdf[isf], fontsize=21)
    ax.set_ylim(yranges[isf][0], yranges[isf][1])
    ax.set_xlabel(r"$Q~({\rm GeV})$", fontsize=20)
    ax.text(0.05, 0.85, r"$x=0.794$", fontsize=16, transform=ax.transAxes)

    ax.legend(
        [(p1[0], p2[0]), (p3[0], p4[0]), (p5[0], p6[0])],
        [pdfsetlab[0], pdfsetlab[1], pdfsetlab[2]],
        frameon="True",
        loc=1,
        prop={"size": 14},
    )

py.tight_layout(pad=1, w_pad=1, h_pad=1.0)
py.savefig("StructureFunction" + filelabel + ".pdf")
print("output plot: StructureFunction" + filelabel + ".pdf")

exit()
