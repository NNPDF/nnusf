# -*- coding: utf-8 -*-
import pytest


@pytest.mark.lhapdf
def test_upper_cut():
    from nnusf.theory.bodek_yang import cuts

    assert cuts.Q2MAX > 1.0
