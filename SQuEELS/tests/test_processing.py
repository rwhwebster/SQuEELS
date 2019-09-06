from __future__ import print_function
import unittest
import numpy as np

import SQuEELS.processing as sqp

class TestQuantify(unittest.TestCase):
    def test_quad_glitch_cor_caseA(self):
        nom_corr = np.ones(2048)
        glitch = np.ones(2048)
        glitch[:1024] += 0.1
        glitch[1024:] -= 0.1
        corr = sqp.remedy_quadrant_glitch(glitch, gc=1024, width=10, plot=False)

        assert np.allclose(corr, nom_corr)

    def test_quad_glitch_cor_caseB(self):
        nom_corr = np.ones(2048)
        glitch = np.ones(2048)
        glitch[:1024] -= 0.1
        glitch[1024:] += 0.1
        corr = sqp.remedy_quadrant_glitch(glitch, gc=1024, width=10, plot=False)

        assert np.allclose(corr, nom_corr)


if __name__ == '__main__':
    unittest.main()
