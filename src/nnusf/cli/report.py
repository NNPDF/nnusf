# -*- coding: utf-8 -*-
import pathlib

import click

from ..reports import genhtml
from . import base


@base.command.group("report")
def subcommand():
    """Generate fit reports"""


@click.argument(
    "fitfolder", type=click.Path(exists=True, path_type=pathlib.Path)
)
@subcommand.command("generate")
def sub_generate(fitfolder):
    """Call the main function the generates the report. It takes the
    fit folder as input."""
    genhtml.main(fitfolder)
