# -*- coding: utf-8 -*-

from . import base


@base.command.group("data")
def subcommand():
    """Provide data management utilities."""
