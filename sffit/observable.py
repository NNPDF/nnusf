
from dataclasses import dataclass
import numpy as np

import temp_data


@dataclass
class ObsInfo:
    fktable: np.ndarray = None # output basis to N observables
    xin: np.ndarray = None     # array of length N
    Q2in: np.ndarray = None    # array of length N
    Ain: np.ndarray = None     # array of length N
    


example_fktable = np.random.rand(5,3)
example_xin = np.random.rand(3)
example_Q2in = np.random.rand(3)
example_Ain = np.random.randint(3)

example_fkinfo = ObsInfo(fktable=example_fktable, 
                        xin=example_xin,
                        Q2in=example_Q2in,
                        Ain=example_Ain,)

class Observable:

    def __init__(self, fktable_object) -> None:
        