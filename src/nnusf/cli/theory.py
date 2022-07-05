# -*- coding: utf-8 -*-
"""Provide theory subcommand."""
import logging
import pathlib

import click

from ..theory import grids, predictions, runcards, bodek_yang, compare_to_data
from . import base

_logger = logging.getLogger(__name__)

DESTINATION = pathlib.Path.cwd().absolute() / "theory"
"""Default destination for generated files"""

option_dest = click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=DESTINATION,
    help="Alternative destination path to store the resulting table (default: $PWD/theory)",
)


@base.command.group("theory")
def subcommand():
    """Compute and compare predictions.

    Compute yadism values for structure functions (given an external PDF set)
    and compare with them.
    """
    pass


@subcommand.group("runcards")
def sub_runcards():
    """Generate yadism runcards.

    Dump runcards compatible with predictions.

    """


@sub_runcards.command("by")
@click.option(
    "-u",
    "--theory-update",
    help="String representation of a dictionary containing update for the theory.",
)
@option_dest
def sub_sub_by(theory_update, destination):
    """Bodek-Yang predictions, made with Genie."""
    if theory_update is not None:
        theory_update = eval(theory_update)

    runcards.by(theory_update=theory_update, destination=destination)


@sub_runcards.command("hiq")
@click.argument("data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@option_dest
def sub_sub_hiq(data, destination):
    """High Q2, from cut values of the dataset."""
    runcards.hiq(data, destination=destination)

@sub_runcards.command("all")
@click.argument("data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@option_dest
def sub_sub_all(data, destination):
    """Full datasets runcards"""
    runcards.dvst(data, destination=destination)

@subcommand.command("by")
@click.argument(
    "observables", nargs=-1, type=click.Choice(bodek_yang.load.load().members)
)
@click.option("-a", "--action", multiple=True, type=click.Choice(["npy", "txt"]))
@option_dest
def sub_by(observables, action, destination):
    """Genie's Bodek-Yang output inspection."""

    values, labels = bodek_yang.extract(observables)
    _logger.info(f"Extracted {labels} from Genie data, shape={values.shape}")

    if "txt" in action:
        bodek_yang.dump_text(values, labels=labels, destination=destination)
    if "npy" in action:
        bodek_yang.dump(values, destination=destination)


@subcommand.command("grids")
@click.argument("runcards", type=click.Path(exists=True, path_type=pathlib.Path))
@option_dest
def sub_grids(runcards, destination):
    """Generate grids with yadism.

    RUNCARDS is a path to folder (or tar folder) containing the runcards:
    - only one theory card is expected, whose name has to be `theory.yaml`
    - several observable cards might be provided

    The exact name of the observable cards files are mostly ignored but for
    prefix and suffix: it has to start with `obs`, and extension has to be
    `.yaml`.
    The internal `name` key is used for the generated grids.

    """
    grids.main(runcards.absolute(), destination)


@subcommand.command("predictions")
@click.argument("grids", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("pdf")
@click.option(
    "--err",
    type=click.Choice(["pdf", "theory"], case_sensitive=False),
    default="theory",
)
@click.option("-x", type=int, default=None)
@click.option("--interactive", is_flag=True)
@option_dest
def sub_predictions(grids, pdf, err, destination, x, interactive):
    """Generate predictions from yadism grids.

    GRIDS is a path to folder (or tar folder) containing the grids, one per
    observable.
    PDF is the pdf to be convoluted with the grids, in order to obtain the
    structure functions predictions.

    """
    predictions.main(
        grids.absolute(),
        pdf,
        err=err,
        xpoint=x,
        interactive=interactive,
        destination=destination,
    )

@subcommand.command("compare_to_data")
@click.argument("grids", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("data",  type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("pdf")
@click.option(
    "--err",
    type=click.Choice(["pdf", "theory"], case_sensitive=False),
    default="pdf",
)
@click.option("--interactive", is_flag=True)
@option_dest
def theory_predictions(grids, data, pdf, err, destination, interactive):
    """Generate predictions from yadism grids and compare with data.

    GRIDS is a path to folder (or tar folder) containing the grids, one per
    observable.
    PDF is the pdf to be convoluted with the grids, in order to obtain the
    structure functions predictions.

    """
    compare_to_data.main(
        grids.absolute(),
        data.absolute(),
        pdf,
        err=err,
        interactive=interactive,
        destination=destination,
    )
