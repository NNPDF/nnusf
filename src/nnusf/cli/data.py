# -*- coding: utf-8 -*-
"""Provide data subcommand."""
import pathlib

import click

from ..data import coefficients, combine_tables, filters
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
def filter_all_data(data):
    """Filter the raw dataset.

    Do it alltogether at the same time and dump the resulting Pandas objects
    into the commondata folder.

    The command is run as follows:

        nnu data filter commondata/rawdata/*

    """
    filters.main(data)


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
