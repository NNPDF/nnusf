# -*- coding: utf-8 -*-
import pathlib

from nnusf.data.loader import Loader

path_to_commondata = pathlib.Path(__file__).parents[1].joinpath("commondata")
path_to_coefficients = pathlib.Path(__file__).parents[1].joinpath("coefficients")


class TestLoader:
    def test_init(self):
        data = Loader("NUTEV_F2", path_to_commondata)

        assert data.kinematics.shape[1] == 3
        assert data.covmat.shape == (data.n_data, data.n_data)

    def test_drop_zeros(self):
        data = Loader("BEBCWA59_F3", path_to_commondata)

        assert 0 not in data.table["stat"] + data.fulltables["syst"]

    def test_coefficients_load(self):
        data = Loader("CHORUS_F2", path_to_commondata, path_to_coefficients)

        assert data.coefficients[0].sum() == 1.0
        assert data.coefficients.sum() == data.n_data
        assert data.coefficients.T[[1, 2, 4, 5]].sum() == 0
