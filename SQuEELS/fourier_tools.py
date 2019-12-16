from __future__ import print_function

import numpy as np

from numpy.fft import fft, ifft

from .utils import match_spectra_sizes, extract_ZLP


def fourier_ratio_convolution(HL, LL, deconv, pad=True, ZLPkwargs=None, padkwargs=None):
    '''
    Function for performing Fourier-Ratio deconvolution using a zero-loss
    modifier.  Created because it is unclear what the deconvolution 
    methods in hyperspy are doing.

    Paramters
    ---------
    HL : Hyperspy signal
        The core-loss spectrum to be deconvolved
    LL : Hyperspy signal
        The low-loss spectrum to use for the deconvolution function
    deconv : Boolean
        If true, uses low-loss spectrum to deconvolve high-loss,
        else, convolves high-loss with low-loss
    pad : Boolean
        If True, pads the high-loss end of the spectra with a decay to zero
    ZLPkwargs : dict
        The arguments to be passed to the call for extract_ZLP.
    padkwargs : dict
        Arguments to be passed to match_spectra_sizes

    Returns
    -------
    conv : hyperspy signal
        The forward-convolved core-loss spectrum.
    '''
    # Make copies of data to manipulate
    low = LL.copy()
    high = HL.copy()
    conv = HL.deepcopy()
    sigDim = HL.axes_manager.signal_indices_in_array[0]
    # Record high-loss offset, as this is lost during convolution
    HL_offset = HL.axes_manager[sigDim].offset
    HL_size = HL.axes_manager[sigDim].size
    # Pad spectra for size-matching and continuity at boundaries
    if pad:
        if padkwargs is None:
            padkwargs = {}
        low, high = match_spectra_sizes(LL, HL, **padkwargs)
    # Extract the Zero-loss Peak for the modifier
    if ZLPkwargs is None:
        ZLPkwargs = {}
    ZLP = extract_ZLP(low, **ZLPkwargs)
    # Calculate Fourier Transforms.
    LLF = fft(low.data, axis=sigDim)
    HLF = fft(high.data, axis=sigDim)
    ZLF = fft(ZLP.data, axis=sigDim)
    # Compute Convolution
    if deconv:
        convF = (HLF / LLF) * ZLF
    else:
        convF = (HLF * LLF) / ZLF
    # Extract real part of the inverse transform to get convolved signal back
    iconv = np.real(ifft(convF, axis=sigDim))
    # Restore high-loss spectrum dimensions
    conv.data = iconv[...,:HL_size]

    return conv
