# -*- coding: utf-8 -*-
"""Provide theory subcommand."""
import logging
import pathlib

import click

from ..theory import bodek_yang, compare_to_data, grids, predictions, runcards
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
@click.option(
    "-o",
    "--obs-update",
    help="String representation of a dictionary containing update for the observable description.",
)
@option_dest
def sub_sub_by(theory_update, obs_update, destination):
    """Bodek-Yang predictions, made with Genie."""
    if theory_update is not None:
        theory_update = eval(theory_update)
    if obs_update is not None:
        obs_update = eval(obs_update)

    runcards.by(
        theory_update=theory_update,
        obs_update=obs_update,
        destination=destination.absolute(),
    )


@sub_runcards.command("hiq")
@click.argument(
    "data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)
@option_dest
def sub_sub_hiq(data, destination):
    """High Q2, from cut values of the dataset."""
    runcards.hiq(data, destination=destination)


@sub_runcards.command("all")
@click.argument(
    "data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)
@option_dest
def sub_sub_all(data, destination):
    """Full datasets runcards"""
    runcards.dvst(data, destination=destination, activate_scale_var=False)


@sub_runcards.command("th_err")
@click.argument(
    "data", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)
@option_dest
def sub_sub_th_err(data, destination):
    """Full datasets runcards with 7 points perescription scale variations"""
    runcards.dvst(data, destination=destination, activate_scale_var=True)


@subcommand.command("by")
@click.argument(
    "observables", nargs=-1, type=click.Choice(bodek_yang.load.load().members)
)
@click.option(
    "-a", "--action", multiple=True, type=click.Choice(["npy", "txt"])
)
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
@click.argument(
    "runcards", type=click.Path(exists=True, path_type=pathlib.Path)
)
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
    grids.main(runcards.absolute(), destination.absolute())


@subcommand.command("predictions")
@click.argument("grids", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("pdf")
@click.option(
    "--err",
    type=click.Choice(["pdf", "theory", "combined"], case_sensitive=False),
    default="pdf",
)
@click.option("-x", type=int, default=None)
@click.option("--interactive", is_flag=True)
@click.option(
    "--compare_to_by/--no-compare_to_by",
    default=True,
    help="Compare or not to the Bodek-Yang results.",
)
@option_dest
def sub_predictions(
    grids, pdf, err, destination, x, compare_to_by, interactive
):
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
        destination=destination.absolute(),
        compare_to_by=compare_to_by,
    )


@subcommand.command("compare_to_data")
@click.argument("grids", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("data", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("pdf")
@click.option(
    "--err",
    type=click.Choice(["pdf", "theory"], case_sensitive=False),
    default="pdf",
)
@click.option("--interactive", is_flag=True)
@option_dest
def sub_compare_to_data(grids, data, pdf, err, destination, interactive):
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


@sub_runcards.command("yadknots")
@click.option(
    "-x",
    "--x_grids",
    default=None,
    help="""Stringified dictionary containing specs for x-grid"""
    """" e.g. '{"min": 1e-5, "max": 1.0, "num": 25}'.""",
)
@click.option(
    "-q",
    "--q2_grids",
    default=None,
    help="""Stringified dictionary containing specs for Q2-grid"""
    """" e.g. '{"min": 9e2, "max": 1.96e8, "num": 30}'.""",
)
@click.option(
    "-a",
    "--a_value",
    type=int,
    default=1,
    help="""Value of the Atomic mass number. Default: 1""",
)
@option_dest
def sub_sub_yadknots(x_grids, q2_grids, a_value, destination):
    """Generate runcard at fixed A"""
    if x_grids is not None:
        x_grids = eval(x_grids)
    else:
        x_grids = dict(min=1e-5, max=1.0, num=25)

    if q2_grids is not None:
        q2_grids = eval(q2_grids)
    else:
        q2_grids = dict(min=9e2, max=1.96e8, num=30)

    runcards.yknots(x_grids, q2_grids, a_value, destination)
