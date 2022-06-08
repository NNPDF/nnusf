import numpy as np

# LOAD USING NNUSF COMMONDATA
from nnusf.loader import Loader

import pathlib

path_to_commondata = pathlib.Path(__file__).parents[1].joinpath("commondata")
path_to_theory = pathlib.Path(__file__).parents[1].joinpath("theory")
# data = Loader(path_to_commondata, path_to_theory, "CHORUS", "F3")
data = Loader(path_to_commondata, path_to_theory, "NUTEV", "F2")

input_central_values = data.central_values

input_x = data.kinematics[0]
input_Q2 = data.kinematics[1]
input_A = data.kinematics[2]
input_kinematics_array = np.vstack((input_x, input_Q2, input_A)).T

input_covmat = data.covmat


# MANUALLY ENTERED DATA

ndat = input_central_values.size
input_theory_grid = np.array([[1, 0, 0, 0, 0, 0] for _ in input_central_values])
tr_ratio = 0.75
max_epochs = 2000
patience = 0.9
