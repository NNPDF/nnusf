import functools

import h5py
import numpy as np

from . import utils


class Data:
    def __init__(self, data: h5py.File):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    @property
    def members(self) -> list[str]:
        return list(self.data)

    def __getitem__(self, key) -> np.ndarray:
        return self.dataset(key)[:]

    def dataset(self, key: str) -> h5py.Dataset:
        if isinstance(key, int):
            ds = self.data[self.members[key]]
        else:
            ds = self.data[key]

        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"Only suitable for `Dataset`, while {key} is {type(ds)}")

        return ds


@functools.cache
def load() -> Data:
    genie = h5py.File(utils.pkg / "genie.hdf5", "r")

    return Data(genie)


if __name__ == "__main__":
    d = load()
    __import__("pdb").set_trace()
