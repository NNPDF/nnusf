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
@click.option(
    "-t", "--title", default=None, type=str, help="Title of the report."
)
@click.option(
    "-a", "--author", default=None, type=str, help="Name of the author."
)
@click.option("-k", "--keywords", default=None, type=str, help="Some keywords.")
@subcommand.command("generate")
def sub_generate(fitfolder, title, author, keywords):
    """Call the main function the generates the report. It takes the
    fit folder as input."""
    metadata = {
        "report_title": title,
        "author_name": author,
        "keyword_notes": keywords,
    }
    genhtml.main(fitfolder, **metadata)
