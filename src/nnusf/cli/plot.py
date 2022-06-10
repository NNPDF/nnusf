# -*- coding: utf-8 -*-
"""Provide plot subcommand."""
import pathlib

import click

from ..plot import covmat
from . import base


@base.command.group("plot")
def subcommand():
    """Provide plot utilities."""


@subcommand.command("covmat")
@click.argument("data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "plots",
    help="Alternative destination path to store the resulting table (default: $PWD/plots).",
)
@click.option(
    "-i", "--inverse", is_flag=True, help="Use inverse covariance matrix instead."
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
    help="""Stringified dictionary of cuts, e.g. '{"Q2": {"min": 3.5}}'.""",
)
@click.option(
    "-l", "--symlog", is_flag=True, help="Plot in symmetric logarithmic scale."
)
def sub_combine(data, destination, inverse, norm, cuts, symlog):
    """Combine data tables into a unique one.

    The operation is repeated for each DATA path provided (multiple values allowed),
    e.g.:

        nnu plot covmat commondata/data/*

    to repeat the operation for all datasets stored in `data`.

    """
    if cuts is not None:
        cuts = eval(cuts)

    covmat.main(data, destination, inverse=inverse, norm=norm, cuts=cuts, symlog=symlog)
