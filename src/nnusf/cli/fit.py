# -*- coding: utf-8 -*-
"""Provide fit subcommand."""
import os
import pathlib

import click

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ..sffit import postfit, run_sffit
from . import base


@base.command.group("fit")
def subcommand():
    """Fit structure functions."""


@subcommand.command("run")
@click.argument("runcard", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("replica", type=int)
@click.option(
    "-s",
    "--scan_hyperopt",
    type=int,
    default=None,
    help="Perform a scan of the hyperparameter space using the TPE algorithm."
    "The argument requires the total number of trials for a given replica."
    "eg: nnut fit <runcard.yml> <replica_id> -s 100",
)
@click.option(
    "-d",
    "--destination",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Alternative destination path to store the resulting model"
    "eg: nnu fit run <runcard.yml> <replica_id> -d <destination>"
    "fit model (default: $PWD/commondata)",
)
def sub_run(runcard, replica, scan_hyperopt, destination):
    """Call the sffit run function."""
    run_sffit.main(
        runcard, replica, nbtrials=scan_hyperopt, destination=destination
    )


@subcommand.command("postfit")
@click.argument("model", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-t",
    "--threshold",
    default=None,
    help="""Stringified dictionary containing chis threshold"""
    """" e.g. '{"tr_max": 5, "vl_max": 5}'.""",
)
def sub_postfit(model, threshold):
    """Perform a postfit on a fit folder by discarding the replica
    that does satisfy some criteria.
    """
    if threshold is not None:
        threshold = eval(threshold)
    postfit.main(model, chi2_threshold=threshold)
