# -*- coding: utf-8 -*-
import numpy as np

Q2MIN = 1.5**2
Q2MAX = 15**2
XMIN = 1e-3

xcut = lambda x: XMIN < x
q2cut = lambda q2: np.logical_and(Q2MIN < q2, q2 < Q2MAX)
