from __future__ import print_function
import unittest
import numpy as np

import SQuEELS.quantify as sqq

class TestQuantify(unittest.TestCase):
    def test_normalEqn_univariate(self):
        nDat = 10
        nom_theta = np.array([0.5, 2.0])
        X1 = np.arange(nDat)
        X0 = np.ones(nDat)
        Y = nom_theta[0] + nom_theta[1]*X1 
        X = np.vstack((X0, X1)).T

        theta = sqq._normalEqn(X, Y)

        assert np.allclose(theta, nom_theta)

    def test_normalEqn_multivariate(self):
        # TODO Update Test
        nDat = 10
        nom_theta = np.array([0.5, 2.0])
        X1 = np.arange(nDat)
        X0 = np.ones(nDat)
        Y = nom_theta[0] + nom_theta[1]*X1 
        X = np.vstack((X0, X1)).T

        theta = sqq._normalEqn(X, Y)

        assert np.allclose(theta, nom_theta)



if __name__ == '__main__':
    unittest.main()
