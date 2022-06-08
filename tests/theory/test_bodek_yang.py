# -*- coding: utf-8 -*-
from nnusf.theory.bodek_yang import cuts


def test_upper_cut():
    assert cuts.Q2MAX > 1.0
