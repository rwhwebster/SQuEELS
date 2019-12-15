from __future__ import print_function

import numpy as np

from numpy.fft import fft, ifft

from .utils import match_spectra_sizes, extract_ZLP


def fourier_ratio_convolution(HL, LL, deconv, pad=True, ZLPkwargs=None):
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
    deconv : Boolean
        If true, uses low-loss spectrum to deconvolve high-loss,
        else, convolves high-loss with low-loss
    pad : Boolean
        If True, pads the high-loss end of the spectra with a decay to zero
    ZLPkwargs : dict
        The arguments to be passed to the call for extract_ZLP.

    Returns
    -------
    reconv : hyperspy spectrum object
        The forward-convolved core-loss spectrum.
    '''
    # Make copies of data to manipulate
    low = LL.copy()
    high = HL.copy()
    conv = HL.deepcopy()
    # Record high-loss offset, as this is lost during convolution
    HL_offset = HL.axes_manager[0].offset
    HL_size = HL.axes_manager[0].size
    # Pad spectra for size-matching and continuity at boundaries
    if pad:
        low, high = match_spectra_sizes(LL, HL)
    if ZLPkwargs is None:
        ZLPkwargs = {}
    # Extract the Zero-loss Peak for the modifier
    ZLP = extract_ZLP(low, **ZLPkwargs)
    # Calculate Fourier Transforms.
    LLF = fft(low.data)
    HLF = fft(high.data)
    ZLF = fft(ZLP.data)
    # Compute Convolution
    if deconv:
        convF = (HLF / LLF) * ZLF
    else:
        convF = (HLF * LLF) / ZLF
    # Extract real part of the inverse transform to get convolved signal back
    iconv = np.real(ifft(convF))
    # Restore high-loss spectrum dimensions
    conv.data = iconv[:HL_size]

    return conv
