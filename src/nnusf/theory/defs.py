# -*- coding: utf-8 -*-
"""Physics constants and definitions."""

import itertools

sfmap = dict(F2="F2_total", FL="FL_total", F3="F3_total")
xsmap = dict(
    CHORUS="XSCHORUSCC", NUTEV="XSNUTEVNU", CDHSW="XSCHORUSCC", FW="FW"
)
projectiles = dict(NU="neutrino", NB="antineutrino")
targets = {20: "neon", 100: "marble", 56: "iron", 208: "lead"}

three_points = [0.5, 1.0, 2.0]
"Three points prescription for scale variations."

nine_points = list(itertools.product(three_points, three_points))
"""Nine points prescription for scale variations (as couples, referred to ``(fact,
ren)`` scales)."""
