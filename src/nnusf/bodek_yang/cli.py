import pathlib

import click

from . import grids, predictions, runcards
from ..cli import base


@base.command.group("by")
def subcommand():
    """Compare predictions for Bodek-Yang model.

    Compute yadism values for structure functions (given an external PDF set)
    and compare with them.
    """
    pass


@subcommand.command("runcards")
def sub_runcards():
    """Generate yadism runcards.

    Dump runcards compatible with Genie predictions.
    """
    runcards.main()


@subcommand.command("grids")
@click.argument("runcards", type=click.Path(exists=True, path_type=pathlib.Path))
def sub_grids(runcards):
    """Generate grids with yadism.

    RUNCARDS is a path to folder (or tar folder) containing the runcards:
    - only one theory card is expected, whose name has to be `theory.yaml`
    - several observable cards might be provided

    The exact name of the observable cards files are mostly ignored but for
    prefix and suffix: it has to start with `obs`, and extension has to be
    `.yaml`.
    The internal `name` key is used for the generated grids.
    """
    grids.main(runcards.absolute())


@subcommand.command("predictions")
@click.argument("grids", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("pdf")
@click.option(
    "--err",
    type=click.Choice(["pdf", "theory"], case_sensitive=False),
    default="theory",
)
def sub_predictions(grids, pdf, err):
    """Generate predictions from yadism grids.

    GRIDS is a path to folder (or tar folder) containing the grids, one per
    observable.
    PDF is the pdf to be convoluted with the grids, in order to obtain the
    structure functions predictions.
    """
    predictions.main(grids.absolute(), pdf, err=err)
