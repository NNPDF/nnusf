import numpy as np

ndat = 120

input_central_values = np.random.rand(1,ndat)

input_Q2 = np.random.rand(1,ndat)*5
input_x = np.random.rand(1,ndat)
input_A = np.random.randint(low=1, high=100, size=(1,ndat))
input_array = np.concatenate((input_x, input_Q2, input_A), axis=0).T

input_covmat = np.random.rand(ndat,ndat)

input_theory_grid = np.array([[1,0,0,0,0,0] for _ in input_central_values[0]])
