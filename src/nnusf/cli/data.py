# -*- coding: utf-8 -*-
import pathlib

import click

from ..data import coefficients, combine_tables
from ..data.filters import (
    filter_bebcwa59,
    filter_chorus,
    filter_nutev,
    filter_charm,
    filter_cdhsw,
    filter_ccfr,
)
from . import base


@base.command.group("data")
def subcommand():
    """Provide data management utilities."""


@subcommand.command("combine")
@click.argument("data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "commondata",
    help="Alternative destination path to store the resulting table (default: $PWD/commondata)",
)
def sub_combine(data, destination):
    """Combine data tables into a unique one.

    The operation is repeated for each DATA path provided (multiple values allowed),
    e.g.:

        nnu data coefficients commondata/data/*

    to repeat the operation for all dataset stored in `data`.
    """
    combine_tables.main(data, destination)


@subcommand.command("filter")
@click.argument("data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("name", nargs=1, type=str)
def filter_data(data, name):
    if name == "BEBCWA59":
        filter_bebcwa59.main(data)
    elif name == "CHORUS":
        filter_chorus.main(data)
    elif name == "NUTEV":
        filter_nutev.main(data)
    elif name == "CHARM":
        filter_charm.main(data)
    elif name == "CCFR":
        filter_ccfr.main(data)
    elif name == "CDHSW":
        filter_cdhsw.main(data)
    else:
        raise ValueError("Dataset not Implemented.")


@subcommand.command("coefficients")
@click.argument("data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "coefficients",
    help="Alternative destination path to store the coefficients (default: $PWD/coefficients)",
)
def sub_coefficients(data, destination):
    """Provide coefficients for the observables.

    Dump coefficients to connect the structure functions basis (F2, FL, and F3)
    to the given experimental observable.

    The operation is repeated for each DATA path provided (multiple values allowed),
    e.g.:

        nnu data coefficients commondata/data/*

    to repeat the operation for all dataset stored in `data`.
    """
    coefficients.main(data, destination)
