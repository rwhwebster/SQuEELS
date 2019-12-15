from __future__ import print_function
import unittest
import numpy as np

import SQuEELS.utils as squ

class TestUtils(unittest.TestCase):
    def test_1d_hann_window(self):
        window = squ.generate_hann_window((1000,))

    def test_2d_hann_window(self):
        window_x = squ.generate_hann_window((1000,1000), direction=0)
        window_y = squ.generate_hann_window((1000,1000), direction=1)

    def test_3d_hann_window(self):
        window_z = squ.generate_hann_window((100,100,1000), direction=2)


if __name__ == '__main__':
    unittest.main()
