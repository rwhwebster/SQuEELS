from __future__ import print_function
import unittest
import os
import zipfile
import tempfile
import shutil
import numpy as np
import hyperspy.api as hs

import SQuEELS.io as sqio
import SQuEELS.bayes as sqb

class TestBayes(unittest.TestCase):
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

    def test_model_creation(self):
        lib = sqio.Standards(fp=self.ref_dir)

        SI = sqio.Data(fp=os.path.join(self.dat_dir, 'deconv.dm3'))

        model = sqb.BayesModel(SI, lib, ['Ti'], (400.0, 480.0))

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.ref_dir)
        shutil.rmtree(self.dat_dir)

if __name__ == '__main__':
    unittest.main()
