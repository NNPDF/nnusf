import pathlib

import click

from . import grids, runcards


@click.group()
def command():
    pass


@command.command("runcards")
def sub_runcards():
    """Generate Yadism runcards.

    Dump runcards compatible with Genie predictions.
    """
    runcards.main()


@command.command("grids")
@click.argument("runcards", type=click.Path(exists=True, path_type=pathlib.Path))
def sub_grids(runcards):
    """Generate grids with Yadism.

    RUNCARDS is a path to folder (or tar folder) containing the runcards:
    - only one theory card is expected, whose name has to be `theory.yaml`
    - several observable cards might be provided

    The exact name of the observable cards files are mostly ignored but for
    prefix and suffix: it has to start with `obs`, and extension has to be
    `.yaml`.
    The internal `name` key is used for the generated grids.
    """
    grids.main(runcards.absolute())
