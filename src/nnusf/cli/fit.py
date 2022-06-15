# -*- coding: utf-8 -*-
"""Provide fit subcommand."""
import os
import pathlib

import click

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ..sffit import run_sffit
from . import base


@base.command.group("fit")
def subcommand():
    """Fit structure functions."""


@subcommand.command("run")
@click.argument("runcard", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("-r", "--replica", type=int, default=1)
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "fits",
    help="Alternative destination path to store the resulting model (default: $PWD/commondata)",
)
def sub_run(runcard, replica, destination):
    run_sffit.main(runcard, replica, destination=destination)
