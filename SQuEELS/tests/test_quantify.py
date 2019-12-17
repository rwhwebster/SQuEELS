from __future__ import print_function
import unittest
import os
import zipfile
import tempfile
import shutil
import numpy as np

import SQuEELS.quantify as sqq
import SQuEELS.io as sqio

class TestQuantify(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ref_dir = tempfile.mkdtemp(prefix='sq-')
        self.ref_bp = os.path.dirname(__file__)

        ref_fn = os.path.join(self.ref_bp, 'test_data', 'ref_data.zip')
        with zipfile.ZipFile(ref_fn, 'r') as z:
            z.extractall(path=self.ref_dir)
            ref_fns = z.namelist()
        self.ref_fns = [x for x in ref_fns if x.split('.')[-1].lower() in ['dm3']]
        self.ref_fns.sort()

        self.dat_dir = tempfile.mkdtemp(prefix='sq-')
        self.dat_bp = os.path.dirname(__file__)

        dat_fn = os.path.join(self.dat_bp, 'test_data', 'test_data.zip')
        with zipfile.ZipFile(dat_fn, 'r') as z:
            z.extractall(path=self.dat_dir)
            dat_fns = z.namelist()
        self.dat_fns = [x for x in dat_fns if x.split('.')[-1].lower() in ['dm3']]
        self.dat_fns.sort()

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

    def test_MLLSmodel_creation(self):
        lib = sqio.Standards(fp=self.ref_dir)
        SI = sqio.Data(fp=os.path.join(self.dat_dir, 'deconv.dm3'))
        model = sqq.MLLSmodel(SI, lib)

    def test_MLLSmodel_point(self):
        lib = sqio.Standards(fp=self.ref_dir)
        SI = sqio.Data(fp=os.path.join(self.dat_dir, 'deconv.dm3'))
        SI.set_data_range(400.0, 500.0)
        tofit = ['Ti',]
        lib.set_spectrum_range(400.0, 500.0)
        model = sqq.MLLSmodel(SI, lib)


if __name__ == '__main__':
    unittest.main()
