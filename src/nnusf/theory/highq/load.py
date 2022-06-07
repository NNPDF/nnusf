# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

sfmap = {"F2": "F2_total", "FL": "FL_total", "F3": "F3_total"}
xsmap = {"CHORUS": "XSCHORUSCC"}
Q2CUT = 5

xgrid = np.geomspace(1e-4, 1, 20)
q2grid = np.geomspace(Q2CUT, 1e4, 20)


def kinematics(n=1000) -> pd.DataFrame:
    kins = dict(
        x=np.random.random(n),
        Q2=np.exp(np.random.random(n) * 3 + 1),
        y=np.random.random(n),
        obs=np.random.choice(list(sfmap.keys()) + ["XS"], size=n),
        proj=np.random.randint(-1, 2, size=n),
        A=np.random.randint(1, 100, size=n),
    )

    return pd.DataFrame(kins)
