import numpy as np
import pandas as pd


def kinematics(n=1000) -> pd.DataFrame:
    kins = dict(
        x=np.random.random(n),
        Q2=np.exp(np.random.random(n) * 3 + 1),
        y=np.random.random(n),
        obs=np.random.randint(1, 4, size=n),
        proj=np.random.randint(-1, 1, size=n),
        A=np.random.randint(1, 100, size=n),
    )

    return pd.DataFrame(kins)
