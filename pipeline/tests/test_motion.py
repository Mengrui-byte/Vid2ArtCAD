import unittest

import numpy as np

from pipeline.core.motion import quintic_segment


def test_quintic_boundaries_and_monotonicity():
    q, t = quintic_segment(0, 1, 4, 30)
    assert q[0] == 0 and q[-1] == 1
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(np.diff(t) > 0)

class MotionTest(unittest.TestCase):
    def test_quintic(self):
        test_quintic_boundaries_and_monotonicity()
