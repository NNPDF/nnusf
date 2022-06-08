# -*- coding: utf-8 -*-
import pathlib

import click

from ..data import coefficients, combine_tables
from . import base


@base.command.group("data")
def subcommand():
    """Provide data management utilities."""


@subcommand.command("combine")
def sub_combine():
    """Combine tables."""
    combine_tables.main()


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
