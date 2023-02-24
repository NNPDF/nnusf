# -*- coding: utf-8 -*-
"""Provide extra scripts subcommand."""
import click
import numpy as np

from ..scripts import integrate, isoscalar
from . import base


@base.command.group("extra")
def subcommand():
    """Extra commands for various scripts."""


@subcommand.command("integrate")
@click.argument("pdfset", type=str)
@click.option(
    "-p",
    "--pid",
    type=int,
    default=1001,
    help="""Structure function LHAPDF ID. The available options are """
    """[1001, 1002, 1003, 2001, 2002, 2003, 3001, 3002, 3003]"""
    """ Default PID: 1001 (F_2).""",
)
@click.option(
    "-x",
    "--x_grids",
    default=None,
    help="""Stringified dictionary containing specs for x-grid"""
    """" e.g. '{"min": 0.01, "max": 1.0, "num": 100}'.""",
)
@click.option(
    "-q",
    "--q2_grids",
    default=None,
    help="""Stringified dictionary containing specs for Q2-grid"""
    """" e.g. '{"min": 0.001, "max": 100000, "num": 200}'.""",
)
def sub_integrate(pdfset, pid, x_grids, q2_grids):
    """Compute the integral of a Structure Function given a LHAPDF ID.

    nnu extra integrate <pdfset_name>
    """
    if x_grids is not None:
        x_grids = eval(x_grids)
        xgrid = np.geomspace(x_grids["min"], x_grids["max"], num=x_grids["num"])
    else:
        xgrid = None

    if q2_grids is not None:
        q2_grids = eval(q2_grids)
        q2grid = np.geomspace(
            q2_grids["min"], q2_grids["max"], num=q2_grids["num"]
        )
    else:
        q2grid = None

    integrate.main(pdfset, pid, xgrid, q2grid)


@subcommand.command("impose_isoscalar")
@click.argument("pdfset", type=str)
@click.option(
    "--install/--no-install",
    default=True,
    help="Install the set into the LHAPDF directory",
)
@click.option(
    "-a",
    "--a_value",
    type=int,
    default=1,
    help="""Atomic mass number value. Default: 1""",
)
@click.option(
    "--install/--no-install",
    default=True,
    help="Install the set into the LHAPDF directory",
)
def sub_impose_isoscalar(pdfset, a_value, install):
    isoscalar.main(pdfset, a_value, install)
