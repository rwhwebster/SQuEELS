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
        assert expected == list(lib.data.keys())

    def test_active(self):
        lib = sqio.Standards(fp=self.tmp_dir)

        lib.set_active_standards(['Ti', 'V'])

        expected = [False, False, True, True]
        assert expected == list(lib.active.values())

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tmp_dir)



if __name__ == '__main__':
    unittest.main()
