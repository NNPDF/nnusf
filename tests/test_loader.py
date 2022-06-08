import pathlib
from nnusf.data.loader import Loader

path_to_commondata = pathlib.Path(__file__).parents[1].joinpath("commondata")
path_to_theory = pathlib.Path(__file__).parents[1].joinpath("theory")


class TestLoader:
    def test_init(self):
        data = Loader(path_to_commondata, path_to_theory, "NUTEV_F2")

        assert data.kinematics.shape[1] == 3
        assert data.covmat.shape == (data.n_data,  data.n_data)

    def test_drop_zeros(self):
        data = Loader(path_to_commondata, path_to_theory, "BEBCWA59_F3")

        assert 0 not in data.fulltables["stat"] + data.fulltables["syst"]
