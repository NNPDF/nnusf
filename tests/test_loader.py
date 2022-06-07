import pathlib
from nnusf.loader import Loader

path_to_commondata = pathlib.Path(__file__).parents[1].joinpath("commondata")
path_to_theory = pathlib.Path(__file__).parents[1].joinpath("theory")

class TestLoader:
    def test_init(self):
        data = Loader(path_to_commondata, path_to_theory, "NUTEV", "F2")

        assert len(data.kinematics) == 3
        n_data = data.kinematics[0].shape[0]
        assert data.covmat.shape == (n_data, n_data)
