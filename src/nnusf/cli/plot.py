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
    help="Alternative destination path to store the resulting table (default: $PWD/plots)",
)
@click.option("-i", "--inverse", is_flag=True)
@click.option(
    "-c",
    "--cuts",
    default=None,
    help="""stringified dictionary of cuts, e.g. '{"Q2": {"min": 3.5}}'""",
)
def sub_combine(data, destination, inverse, cuts):
    """Combine data tables into a unique one.

    The operation is repeated for each DATA path provided (multiple values allowed),
    e.g.:

        nnu data coefficients commondata/data/*

    to repeat the operation for all dataset stored in `data`.

    """
    if cuts is not None:
        cuts = eval(cuts)

    covmat.main(data, destination, inverse=inverse, cuts=cuts)
