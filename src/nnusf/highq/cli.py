import pathlib

import click

from . import runcards
from ..cli import base


@base.command.group("hiq")
def subcommand():
    """Compute predictions for large Q2.

    Using yadism (again).
    """
    pass


@subcommand.command("runcards")
def sub_runcards():
    """Generate yadism runcards.

    Dump runcards [...].
    """
    runcards.main()
