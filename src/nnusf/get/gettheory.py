"""Provide extra scripts subcommand."""
from pathlib import Path

import click
from appdirs import user_data_dir
from rich.console import Console

from ..scripts import gettheory
from . import base

console = Console()


@base.command.group("get")
def subcommand():
    """Commands to download commondata theory."""


@subcommand.command("theory")
def sub_get_theory():
    """Download the theory and store in user directory."""
    gettheory.main()


@subcommand.command("print_userdir_path")
def sub_print_userdir():
    """Print the user directory path."""
    path = Path(user_data_dir()).joinpath("nnusf")
    console.print(f"NNUSF USERDIR: '{path}'", style="bold red")
