# -*- coding: utf-8 -*-
"""Provide data subcommand."""
import pathlib

import click

from ..data import coefficients, combine_tables, filters, matching_grids
from . import base

destination_path = click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "commondata",
    help="Alternative destination path to store the resulting table (default: $PWD/commondata)",
)

dataset_path = click.argument(
    "data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)


@base.command.group("data")
def subcommand():
    """Provide data management utilities."""


@subcommand.command("combine")
@dataset_path
@destination_path
def sub_combine(data, destination):
    """Combine data tables into a unique one.

    The operation is repeated for each DATA path provided (multiple values allowed),
    e.g.:

        nnu data coefficients commondata/data/*

    to repeat the operation for all dataset stored in `data`.

    """
    combine_tables.main(data, destination)


@subcommand.command("filter")
@dataset_path
def filter_all_data(data):
    """Filter the raw dataset.

    Do it alltogether at the same time and dump the resulting Pandas objects
    into the commondata folder.

    The command is run as follows:

        nnu data filter commondata/rawdata/*

    """
    filters.main(data)


@subcommand.command("coefficients")
@dataset_path
@destination_path
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


@subcommand.command("matching_grids")
@dataset_path
@destination_path
def sub_matching_grids(destination, data):
    """
    Generate fake data for matching with theory
    """
    matching_grids.main(destination, data)


@subcommand.command("proton_bc")
@destination_path
def sub_proton_bc(destination):
    """
    Generate fake data for boundary proton condition
    """
    matching_grids.proton_boundary_conditions(destination)
