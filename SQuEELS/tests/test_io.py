from __future__ import print_function
import unittest
import os
import zipfile
import tempfile
import shutil
import numpy as np

import SQuEELS.io as sqio

class TestStandards(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='sq-')
        self.bp = os.path.dirname(__file__)

        fn = os.path.join(self.bp, 'test_data', 'ref_data.zip')
        with zipfile.ZipFile(fn, 'r') as z:
            z.extractall(path=self.tmp_dir)
            fns = z.namelist()
        self.fns = [x for x in fns if x.split('.')[-1].lower() in ['dm3']]
        self.fns.sort()

    def test_load(self):
        # loads a test directory which contains Cu, Ni, Ti and V
        lib = sqio.Standards(fp=self.tmp_dir)

        expected = ['Cu', 'Ni', 'Ti', 'V']
        actual = list(lib.data.keys())
        actual.sort()
        assert expected == actual

    def test_active(self):
        lib = sqio.Standards(fp=self.tmp_dir)

        lib.set_active_standards(['Ti', 'V'])

        expected = [False, False, True, True]
        actual = []
        for item in ['Cu', 'Ni', 'Ti', 'V']:
            actual.append(lib.active[item])
        assert expected == actual

    def test_range_crop_case_A(self):
        # crop when both limits are within data range
        lib = sqio.Standards(fp=self.tmp_dir)
        lib.set_active_standards(['Ti'])
        lib.set_spectrum_range(500.0, 600.0)

    def test_range_crop_case_B(self):
        # crop when left limit is beyond data range
        lib = sqio.Standards(fp=self.tmp_dir)
        lib.set_active_standards(['Ti'])
        lib.set_spectrum_range(300.0, 600.0)

    def test_range_crop_case_C(self):
        # crop when right limit is beyond data range
        lib = sqio.Standards(fp=self.tmp_dir)
        lib.set_active_standards(['Ti'])
        lib.set_spectrum_range(500.0, 1500.0)

    def test_range_crop_case_D(self):
        # crop when both limits are beyond data range
        lib = sqio.Standards(fp=self.tmp_dir)
        lib.set_active_standards(['Ti'])
        lib.set_spectrum_range(300.0, 1500.0)

    def test_normalisation_case_A(self):
        lib = sqio.Standards(fp=self.tmp_dir)

        lib.set_active_standards(['Ti','Ni'])

        lib.set_spectrum_range(400.0, 1000.0)

        lib.normalise(logscale=False)

    def test_normalisation_case_B(self):
        # log normalisation
        lib = sqio.Standards(fp=self.tmp_dir)

        lib.set_active_standards(['Ti','Ni'])

        lib.set_spectrum_range(400.0, 1000.0)

        lib.normalise(logscale=True)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tmp_dir)



if __name__ == '__main__':
    unittest.main()
