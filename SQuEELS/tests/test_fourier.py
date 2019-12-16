from __future__ import print_function
import unittest
import numpy as np
import scipy as sp
import hyperspy.api as hs

import SQuEELS.fourier_tools as sqf

class TestFourier(unittest.TestCase):
    def test_FR_conv_1d(self):
        det = np.zeros(2048)
        HL_data = det.copy()
        HL_data[200:1000] = 1.0
        head = sp.signal.gaussian(300, std=40)
        tail = sp.signal.gaussian(2000, std=250, sym=False)
        HL_data[-1700:-700] = tail[-1000:]
        HL_data[125:275] = head[:150]
        HL_signal = hs.signals.Signal1D(HL_data)
        LL_data = det.copy()
        LL_data[:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[0].offset = -200
        output = sqf.fourier_ratio_convolution(HL_signal, LL_signal, deconv=False)

    def test_FR_conv_2d(self):
        det = np.zeros((10,2048))
        HL_data = det.copy()
        HL_data[:,200:1000] = 1.0
        head = sp.signal.gaussian(300, std=40)
        tail = sp.signal.gaussian(2000, std=250, sym=False)
        HL_data[:,-1700:-700] = tail[-1000:]
        HL_data[:,125:275] = head[:150]
        HL_signal = hs.signals.Signal1D(HL_data)
        LL_data = det.copy()
        LL_data[:,:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:,:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[1].offset = -200
        output = sqf.fourier_ratio_convolution(HL_signal, LL_signal, deconv=False)

    def test_FR_conv_3d(self):
        det = np.zeros((10,20,2048))
        HL_data = det.copy()
        HL_data[...,200:1000] = 1.0
        head = sp.signal.gaussian(300, std=40)
        tail = sp.signal.gaussian(2000, std=250, sym=False)
        HL_data[...,-1700:-700] = tail[-1000:]
        HL_data[...,125:275] = head[:150]
        HL_signal = hs.signals.Signal1D(HL_data)
        LL_data = det.copy()
        LL_data[...,:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[...,:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[2].offset = -200
        output = sqf.fourier_ratio_convolution(HL_signal, LL_signal, deconv=False)

    def test_FR_deconv_1d(self):
        det = np.zeros(2048)
        HL_data = det.copy()
        HL_data[200:1000] = 1.0
        head = sp.signal.gaussian(300, std=40)
        tail = sp.signal.gaussian(2000, std=250, sym=False)
        HL_data[-1700:-700] = tail[-1000:]
        HL_data[125:275] = head[:150]
        HL_signal = hs.signals.Signal1D(HL_data)
        LL_data = det.copy()
        LL_data[:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[0].offset = -200
        output = sqf.fourier_ratio_convolution(HL_signal, LL_signal, deconv=True)

    def test_FR_deconv_2d(self):
        det = np.zeros((10,2048))
        HL_data = det.copy()
        HL_data[:,200:1000] = 1.0
        head = sp.signal.gaussian(300, std=40)
        tail = sp.signal.gaussian(2000, std=250, sym=False)
        HL_data[:,-1700:-700] = tail[-1000:]
        HL_data[:,125:275] = head[:150]
        HL_signal = hs.signals.Signal1D(HL_data)
        LL_data = det.copy()
        LL_data[:,:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[:,:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[1].offset = -200
        output = sqf.fourier_ratio_convolution(HL_signal, LL_signal, deconv=True)

    def test_FR_deconv_3d(self):
        det = np.zeros((10,20,2048))
        HL_data = det.copy()
        HL_data[...,200:1000] = 1.0
        head = sp.signal.gaussian(300, std=40)
        tail = sp.signal.gaussian(2000, std=250, sym=False)
        HL_data[...,-1700:-700] = tail[-1000:]
        HL_data[...,125:275] = head[:150]
        HL_signal = hs.signals.Signal1D(HL_data)
        LL_data = det.copy()
        LL_data[...,:400] = sp.signal.gaussian(400, std=2, sym=False)
        LL_data[...,:800] += sp.signal.gaussian(800, std=60)/20
        LL_signal = hs.signals.Signal1D(LL_data)
        LL_signal.axes_manager[2].offset = -200
        output = sqf.fourier_ratio_convolution(HL_signal, LL_signal, deconv=True)

if __name__ == '__main__':
    unittest.main()
