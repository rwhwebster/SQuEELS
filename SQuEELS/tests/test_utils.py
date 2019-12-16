from __future__ import print_function
import unittest
import numpy as np
import hyperspy.api as hs

import SQuEELS.utils as squ

class TestUtils(unittest.TestCase):
    def test_1d_hann_window(self):
        window = squ.generate_hann_window((1000,))

    def test_2d_hann_window(self):
        window_x = squ.generate_hann_window((1000,1000), direction=0)
        window_y = squ.generate_hann_window((1000,1000), direction=1)

    def test_3d_hann_window(self):
        window_z = squ.generate_hann_window((100,100,1000), direction=2)

    def test_pad_spectrum(self):
        # Mock spectrum object
        spectrum = hs.signals.Signal1D(2048)
        spectrum.data[:] = 1.0
        # Pass mock spectrum through pad_spectrum
        new_spectrum = squ.pad_spectrum(spectrum, 4096)

    def test_match_spectra_sizes_1d(self):
        s1 = hs.signals.Signal1D(2000)
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(1000)
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_2d(self):
        s1 = hs.signals.Signal1D((10,2000))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D((10,1000))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_3d(self):
        s1 = hs.signals.Signal1D((10,20,2000))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D((10,20,1000))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2)
        assert np.allclose(o1.data.shape, o2.data.shape)


if __name__ == '__main__':
    unittest.main()
