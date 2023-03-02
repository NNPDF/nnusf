# -*- coding: utf-8 -*-
"""Provide fit subcommand."""
import os
import pathlib

import click

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ..lhapdf import dump_grids
from ..sffit import postfit, run_sffit
from . import base


@base.command.group("fit")
def subcommand():
    """Fit structure functions."""


@subcommand.command("run")
@click.argument("runcard", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("replica", type=int)
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute(),
    help="Alternative destination path to store the resulting model (default: $PWD)",
)
def sub_run(runcard, replica, destination):
    """Call the sffit run function."""
    run_sffit.main(runcard, replica, destination=destination)


@subcommand.command("postfit")
@click.argument("model", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-n",
    "--nrep",
    default=None,
    type=int,
    help="""Number of of final replicas""",
)
@click.option(
    "-t",
    "--threshold",
    default=None,
    help="""Stringified dictionary containing chis threshold"""
    """" e.g. '{"tr_max": 5, "vl_max": 5}'.""",
)
def sub_postfit(model, threshold, nrep):
    """Perform a postfit on a fit folder by discarding the replica
    that does satisfy some criteria.
    """
    if threshold is not None:
        threshold = eval(threshold)
    postfit.main(model, chi2_threshold=threshold, ntot_rep=nrep)


@subcommand.command("dump_grids")
@click.argument("model", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-s",
    "--sfset_name",
    default=None,
    help="""Name of the SF LHAPDF set with with the high-Q2 matching will """
    """computed. If not specified, the NNUSF predictions will not be matched.""",
)
@click.option(
    "-a",
    "--a_value",
    type=int,
    default=1,
    help="""Atomic mass number value. Default: 1""",
)
@click.option(
    "-x",
    "--x_grids",
    default=None,
    help="""Stringified dictionary containing specs for x-grid"""
    """" e.g. '{"min": 1e-5, "max": 1.0, "num": 100}'.""",
)
@click.option(
    "-q",
    "--q2_grids",
    default=None,
    help="""Stringified dictionary containing specs for Q2-grid"""
    """" e.g. '{"min": 1e-3, "max": 400, "num": 100}'.""",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default="NNUSF10_Q2MIN001",
    help="Alternative LHAPDF folder name (default: $PWD/NNUSF10_Q2MIN001)",
)
@click.option(
    "-m",
    "--min_highq2",
    default=None,
    help="Minimal value of Q2 for the high-Q2 Yadism predictions.",
)
@click.option(
    "--install/--no-install",
    default=True,
    help="Install the set into the LHAPDF directory",
)
def sub_dump_grids(
    model, sfset_name, a_value, x_grids, q2_grids, output, min_highq2, install
):
    """Generate the LHAPDF grids, dump them into files, and install
    the resulting set into the LHAPDF path.
    """
    if x_grids is not None:
        x_grids = eval(x_grids)
    else:
        x_grids = dict(min=1e-5, max=1.0, num=100)

    if q2_grids is not None:
        q2_grids = eval(q2_grids)
    else:
        q2_grids = dict(min=1e-3, max=400, num=100)

    dump_grids.main(
        model,
        sfset_name,
        a_value,
        x_grids,
        q2_grids,
        output,
        min_highq2,
        install,
    )
