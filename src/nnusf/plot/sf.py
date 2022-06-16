# -*- coding: utf-8 -*-
"""Generate structure functions slices plots."""
import logging
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from .. import utils
from ..data import loader
from . import utils as putils

_logger = logging.getLogger(__file__)


def main(kind, destination):
    print(kind)
    print(destination)
