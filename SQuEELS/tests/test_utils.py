from __future__ import print_function
import unittest
import numpy as np
import scipy as sp
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
        spectrum = hs.signals.Signal1D(np.zeros(2048))
        spectrum.data[:] = 1.0
        # Pass mock spectrum through pad_spectrum
        new_spectrum = squ.pad_spectrum(spectrum, 4096)

    def test_zero_pad_spectrum_1d(self):
        # Mock spectrum object
        spectrum = hs.signals.Signal1D(np.zeros(2048))
        spectrum.data[:] = 1.0
        # Pass mock spectrum through pad_spectrum
        axis = spectrum.axes_manager.signal_indices_in_array[0]
        new_spectrum = squ.zero_pad_spectrum(spectrum, 4096, axis)
        assert np.allclose(new_spectrum.data.shape, (4096,))

    def test_zero_pad_spectrum_2d(self):
        # Mock spectrum object
        spectrum = hs.signals.Signal1D(np.zeros((10,2048)))
        spectrum.data[:] = 1.0
        # Pass mock spectrum through pad_spectrum
        axis = spectrum.axes_manager.signal_indices_in_array[0]
        new_spectrum = squ.zero_pad_spectrum(spectrum, 4096, axis)
        assert np.allclose(new_spectrum.data.shape, (10, 4096))

    def test_zero_pad_spectrum_3d(self):
        # Mock spectrum object
        spectrum = hs.signals.Signal1D(np.zeros((10,10,2048)))
        spectrum.data[:] = 1.0
        # Pass mock spectrum through pad_spectrum
        axis = spectrum.axes_manager.signal_indices_in_array[0]
        new_spectrum = squ.zero_pad_spectrum(spectrum, 4096, axis)
        assert np.allclose(new_spectrum.data.shape, (10, 10, 4096))

    def test_match_spectra_sizes_1d(self):
        s1 = hs.signals.Signal1D(np.zeros(2000))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(np.zeros(1000))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2, taper=False)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_2d(self):
        s1 = hs.signals.Signal1D(np.zeros((10,2000)))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(np.zeros((10,1000)))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2, taper=False)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_3d(self):
        s1 = hs.signals.Signal1D(np.zeros((10,20,2000)))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(np.zeros((10,20,1000)))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2, taper=False)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_1d_with_taper(self):
        s1 = hs.signals.Signal1D(np.zeros(2000))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(np.zeros(1000))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2, taper=True, size=100)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_2d_with_taper(self):
        s1 = hs.signals.Signal1D(np.zeros((10,2000)))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(np.zeros((10,1000)))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2, taper=True, size=100)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_match_spectra_sizes_3d_with_taper(self):
        s1 = hs.signals.Signal1D(np.zeros((10,20,2000)))
        s1.data[:] = 1.0
        s2 = hs.signals.Signal1D(np.zeros((10,20,1000)))
        s2.data[:] = 1.0
        o1, o2 = squ.match_spectra_sizes(s1, s2, taper=True, size=100)
        assert np.allclose(o1.data.shape, o2.data.shape)

    def test_extract_ZLP_1d(self):
        LL_data = np.zeros(2048)
        LL_data[:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[0].offset = -200
        output = squ.extract_ZLP(LL_signal, method='reflected tail', threshold=0.03)

    def test_extract_ZLP_2d(self):
        LL_data = np.zeros((10,2048))
        LL_data[:,:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:,:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[1].offset = -200
        output = squ.extract_ZLP(LL_signal, method='reflected tail', threshold=0.03)

    def test_extract_ZLP_3d(self):
        LL_data = np.zeros((10,20,2048))
        LL_data[:,:,:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:,:,:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[2].offset = -200
        output = squ.extract_ZLP(LL_signal, method='reflected tail', threshold=0.03)

if __name__ == '__main__':
    unittest.main()
