from __future__ import print_function
import unittest
import numpy as np

import absEELS
import absEELS.processing as EELSP

class TestEELSP(unittest.TestCase):

    def test_clip_LL(self):
        # TODO nom_LL = 

        assert np.allclose(nom_LL.data, LL.data)

if __name__ == '__main__':
    unittest.main()
