# -*- coding: utf-8 -*-
"""Provide data subcommand."""
import pathlib

import click

from ..data import coefficients, combine_tables, filters, matching_grids
from . import base

dataset_path = click.argument(
    "data", type=click.Path(exists=True, path_type=pathlib.Path)
)

obs_type = click.argument("obstype", type=str)
pdfset_name = click.argument("pdfset", type=str)

destination_path = click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "commondata",
    help="Alternative destination path (default: $PWD/commondata)",
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

        nnu data combine commondata/data/*

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
@pdfset_name
@destination_path
def sub_matching_grids(data, pdfset, destination):
    """Generate the Yadism data (kinematicas & central values) as
    well as the the predictions for all replicas. The command can
    be run as follows:

    eg: nnu data matching_grids ./grids-CCFR_F2_MATCHING.tar.gz nNNPDF30_nlo_as_0118_A56_Z26
    """
    matching_grids.main(data, pdfset, destination)


@subcommand.command("proton_bc")
@pdfset_name
@obs_type
@destination_path
def sub_proton_bc(pdfset, obstype, destination):
    """Generate the Yadism data (kinematicas & central values) as
    well as the the predictions for all replicas for A=1 use to
    impose the Boundary Condition. The command can be run as follows:

    eg: nnu data proton_bc nNNPDF30_nlo_as_0118_A56_Z26 F2
    """
    matching_grids.proton_boundary_conditions(pdfset, obstype, destination)
