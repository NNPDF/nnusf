import click

from . import runcards


@click.group()
def command():
    pass


@command.command("runcards")
def sub_runcards():
    print(runcards.theory())
