# -*- coding: utf-8 -*-

Q2MIN = 1.5**2
Q2MAX = 15**2
XMIN = 1e-3


def xcut(x):
    return XMIN < x


def q2cut(q2):
    return (Q2MIN < q2) & (q2 < Q2MAX)
