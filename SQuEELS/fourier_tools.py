from __future__ import print_function

import numpy as np

from numpy.fft import fft, ifft

from .processing import extract_ZLP, match_spectra_sizes

import matplotlib.pyplot as plt
plt.ion()


def fourier_ratio_deconvolution(HL, LL, pad=True, ZLPmodel='fit', plot=False):
    '''
    Function for performing Fourier-Ratio deconvolution using a zero-loss
    modifier.  Created because it is unclear what the deconvolution 
    methods in hyperspy are doing.

    Paramters
    ---------
    HL : Hyperspy spectrum
        The core-loss spectrum to be deconvolved
    LL : Hyperspy Spectrum
        The low-loss spectrum to use for the deconvolution function
    pad : Boolean
        If True, pads the high-loss end of the spectra with a decay to zero
    ZLPmodel : string
        The argument to be passed to the call for extract_ZLP to determine how
        the ZLP is extracted from the LL data.
    plot : Boolean
        If true, plots are given of intermediate stages.

    Returns
    -------
    deconv : hyperspy spectrum object
        The deconvolved core-loss spectrum.

    '''
    # Make copies of data to manipulate
    low = LL.copy()
    high = HL.copy()
    reconv = HL.deepcopy()
    # Record high-loss offset, as this is lost during convolution
    HL_offset = HL.axes_manager[0].offset
    HL_size = HL.axes_manager[0].size
    # Pad spectra for size-matching and continuity at boundaries
    if pad:
        low, high = match_spectra_sizes(LL, HL)
    # Extract the Zero-loss Peak for the modifier
    ZLP = extract_ZLP(low, method=ZLPmodel, plot=plot)
    # Calculate Fourier Transforms.
    LLF = fft(low.data)
    HLF = fft(high.data)
    ZLF = fft(ZLP.data)
    # Compute Convolution
    conv = (HLF / LLF) * ZLF
    # Extract real part of the inverse transform to get convolved signal back
    iconv = np.real(ifft(conv))
    # Restore high-loss spectrum dimensions
    deconv.data = iconv[:HL_size]

    return deconv

def reverse_fourier_ratio_convoln(HL, LL, pad=True, ZLPmodel='fit', plot=False):
    '''
    Function for performing Fourier-Ratio deconvolution using a zero-loss
    modifier.  Created because it is unclear what the deconvolution 
    methods in hyperspy are doing.

    Paramters
    ---------
    HL : Hyperspy spectrum
        The core-loss spectrum to be deconvolved
    LL : Hyperspy Spectrum
        The low-loss spectrum to use for the deconvolution function
    pad : Boolean
        If True, pads the high-loss end of the spectra with a decay to zero
    ZLPmodel : string
        The argument to be passed to the call for extract_ZLP to determine how
        the ZLP is extracted from the LL data.
    plot : Boolean
        If true, plots are given of intermediate stages.

    Returns
    -------
    reconv : hyperspy spectrum object
        The forward-convolved core-loss spectrum.
    '''
    # Make copies of data to manipulate
    low = LL.copy()
    high = HL.copy()
    reconv = HL.deepcopy()
    # Record high-loss offset, as this is lost during convolution
    HL_offset = HL.axes_manager[0].offset
    HL_size = HL.axes_manager[0].size
    # Pad spectra for size-matching and continuity at boundaries
    if pad:
        low, high = match_spectra_sizes(LL, HL)
    # Extract the Zero-loss Peak for the modifier
    ZLP = extract_ZLP(low, method=ZLPmodel, plot=plot)
    # Calculate Fourier Transforms.
    LLF = fft(low.data)
    HLF = fft(high.data)
    ZLF = fft(ZLP.data)
    # Compute Convolution
    conv = (HLF * LLF) / ZLF
    # Extract real part of the inverse transform to get convolved signal back
    iconv = np.real(ifft(conv))
    # Restore high-loss spectrum dimensions
    reconv.data = iconv[:HL_size]

    return reconv
