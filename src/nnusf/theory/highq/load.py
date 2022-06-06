# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

obsmap = {"F2": "F2_total", "F3": "F3_total", "XS": "XSCHORUSCC"}


def kinematics(n=1000) -> pd.DataFrame:
    kins = dict(
        x=np.random.random(n),
        Q2=np.exp(np.random.random(n) * 3 + 1),
        y=np.random.random(n),
        obs=np.random.choice(list(obsmap.keys()), size=n),
        proj=np.random.randint(-1, 2, size=n),
        A=np.random.randint(1, 100, size=n),
    )

    return pd.DataFrame(kins)
