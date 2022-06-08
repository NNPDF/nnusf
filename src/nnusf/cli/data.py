# -*- coding: utf-8 -*-
from ..data import coefficients, combine_tables
from . import base


@base.command.group("data")
def subcommand():
    """Provide data management utilities."""


@subcommand.command("combine")
def sub_combine():
    """Combine tables."""
    combine_tables.main()


@subcommand.command("coefficients")
def sub_coefficients():
    """Provide coefficients for the observables.

    Dump coefficients to connect the structure functions basis (F2, FL, and F3)
    to the given experimental observable.
    """
    coefficients.main()
