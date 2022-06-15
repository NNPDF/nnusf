# -*- coding: utf-8 -*-
"""Provide fit subcommand."""
import pathlib

import click

from ..sffit import run_sffit
from . import base


@base.command.group("fit")
def subcommand():
    """Fit structure functions."""

    run_sffit.main()
