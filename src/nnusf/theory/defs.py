# -*- coding: utf-8 -*-
"""Physics constants and definitions."""

import itertools

three_points = [0.5, 1.0, 2.0]
"Three points prescription for scale variations."

nine_points = list(itertools.product(three_points, three_points))
"""Nine points prescription for scale variations (as couples, referred to ``(fact,
ren)`` scales)."""
