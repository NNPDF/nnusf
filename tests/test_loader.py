# -*- coding: utf-8 -*-
from pathlib import Path
from appdirs import user_data_dir
from nnusf.data.loader import Loader

USERDIR = Path(user_data_dir())
path_to_commondata = USERDIR.joinpath("nnusf/commondata")
path_to_coefficients = USERDIR.joinpath("nnusf/coefficients")


class TestLoader:
    def test_init(self):
        data = Loader("NUTEV_F2", path_to_commondata)

        assert data.kinematics.shape[1] == 3
        assert data.covmat.shape == (data.n_data, data.n_data)

    def test_drop_zeros(self):
        data = Loader("BEBCWA59_F3", path_to_commondata)
        combined_unc = data.table["stat"] + data.table["syst"]
        assert 0 not in combined_unc.values

        data_match = Loader("BEBCWA59_F3_MATCHING", path_to_commondata)
        combined_unc_match = data_match.table["stat"] + data_match.table["syst"]
        assert 0 in combined_unc_match.values

    def test_coefficients_load(self):
        data = Loader(
            "CHORUS_F2",
            path_to_commondata=path_to_commondata,
            path_to_coefficients=path_to_coefficients,
        )
        assert data.coefficients[0].sum() == 1.0
        assert data.coefficients.sum() == data.n_data
        assert data.coefficients.T[[1, 2, 4, 5]].sum() == 0

        data_match = Loader(
            "CHORUS_F2_MATCHING",
            path_to_commondata=path_to_commondata,
            path_to_coefficients=path_to_coefficients,
        )
        assert data_match.coefficients[0].sum() == 1.0
        assert data_match.coefficients.sum() == data_match.n_data
        assert data_match.coefficients.T[[1, 2, 4, 5]].sum() == 0
