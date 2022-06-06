import pathlib
from nnusf.loader import Loader

here = pathlib.Path(__file__).parents[1]


class TestLoader:
    def test_init(self):
        data = Loader(here, "NUTEV", "F2")

        assert len(data.kinematics) == 3
        n_data = data.kinematics[0].shape[0]
        assert data.covmat.shape == (n_data, n_data)
