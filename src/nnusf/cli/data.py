# -*- coding: utf-8 -*-
from ..data import combine_tables
from . import base


@base.command.group("data")
def subcommand():
    """Provide data management utilities."""


@subcommand.command("combine")
def sub_combine():
    """Combine tables."""
    combine_tables.main()
