# -*- coding: utf-8 -*-
"""Provide plot subcommand."""
import pathlib

import click

from nnusf.theory.bodek_yang.cuts import q2cut

from ..plot import covmat, fit, kinematics, matching, sf
from ..theory import defs
from . import base


@base.command.group("plot")
def subcommand():
    """Provide plot utilities."""


@subcommand.command("kin")
@click.argument(
    "data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "plots",
    help="Alternative destination path to store the resulting plots (default: $PWD/plots).",
)
@click.option(
    "--wcut/--no-wcut",
    default=True,
    help="Plot line corresponding to W2=3.5 GeV2.",
)
@click.option(
    "--q2max/--no-q2max",
    default=True,
    help="Plot line corresponding to Q2=25 GeV2.",
)
@click.option(
    "--xlog/--no-xlog", default=True, help="Set logarithmic scale on x axis."
)
@click.option(
    "--ylog/--no-ylog", default=True, help="Set logarithmic scale on y axis."
)
@click.option(
    "-c",
    "--cuts",
    default=None,
    help="""Stringified dictionary of cuts, e.g. '{"Q2": {"min": 1.65}, "W2": {"min": 3.5}}'.""",
)
def sub_kinematic(data, destination, wcut, q2max, xlog, ylog, cuts):
    """Generate kinematics plot.

    The plot will include data from each DATA path provided (multiple values allowed),
    to include all of them just run:

        nnu plot kin commondata/data/*

    """
    if cuts is not None:
        cuts = eval(cuts)

    kinematics.main(
        data,
        destination,
        wcut=wcut,
        q2cut=q2max,
        xlog=xlog,
        ylog=ylog,
        cuts=cuts,
    )


@subcommand.command("covmat")
@click.argument(
    "data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "plots",
    help="Alternative destination path to store the resulting plots (default: $PWD/plots).",
)
@click.option(
    "-i",
    "--inverse",
    is_flag=True,
    help="Use inverse covariance matrix instead.",
)
@click.option(
    "-n/-N",
    "--norm/--no-norm",
    default=True,
    help="Normalize covariance matrix with central values (default: True).",
)
@click.option(
    "-c",
    "--cuts",
    default=None,
    help="""Stringified dictionary of cuts, e.g. '{"Q2": {"min": 1.65}, "W2": {"min": 3.5}}'.""",
)
@click.option(
    "-l", "--symlog", is_flag=True, help="Plot in symmetric logarithmic scale."
)
def sub_covmat(data, destination, inverse, norm, cuts, symlog):
    """Generate covariance matrix heatmap.

    The operation is repeated for each DATA path provided (multiple values allowed),
    e.g.:

        nnu plot covmat commondata/data/*

    to repeat the operation for all datasets stored in `data`.

    A further plot will be generated, including the full covariance matrix for
    the union of the datasets selected.

    """
    if cuts is not None:
        cuts = eval(cuts)

    covmat.main(
        data, destination, inverse=inverse, norm=norm, cuts=cuts, symlog=symlog
    )


@subcommand.command("fit")
@click.argument("model", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("runcard", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "plots",
    help="Alternative destination path to store the resulting plots (default: $PWD/plots).",
)
def sub_fit(model, runcard, destination):
    """Plot predictions from the fit and/or compare them to the experimental
    measurements. The command takes two positional arguments and one optional
    argument. The command should be run as follows:

        nnu plot fit <path_fit_folder> <path_plot_yaml> [destination_foder]
    """
    fit.main(model, runcard, destination)


sfkinds = list(defs.sfmap.keys())


@subcommand.command("sf")
@click.argument("dataset", type=click.Path(path_type=pathlib.Path, exists=True))
@click.option(
    "-k",
    "--kind",
    multiple=True,
    type=click.Choice(sfkinds),
    default=sfkinds,
    help="Structure functions kinds to be plotted",
)
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "plots",
    help="Alternative destination path to store the resulting plots (default: $PWD/plots).",
)
def sub_sf(dataset, kind, destination):
    """Plots structure functions."""

    sf.main(dataset, kind, destination)


@subcommand.command("matching_dataset")
@click.argument("dataset", type=click.Path(path_type=pathlib.Path, exists=True))
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "plots",
    help="Alternative destination path to store the resulting plots (default: $PWD/plots).",
)
def sub_matching_dataset(dataset, destination):
    """Plots the matching datasets along with the actual data.

    eg: nnu plot matching_dataset commondata/data/DATA_NUTEV_F2_MATCHING.csv
    """
    matching.main(dataset, destination)
