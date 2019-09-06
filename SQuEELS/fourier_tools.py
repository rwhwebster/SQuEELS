from __future__ import print_function

import numpy as np

from numpy.fft import fft, ifft

from .processing import extract_ZLP, match_spectra_sizes, remove_stray_signal

import matplotlib.pyplot as plt
plt.ion()


def fourier_ratio_deconvolution(HL, LL, stray=False, pad=True, plot=False):
    '''
    Function for performing Fourier-Ratio deconvolution using a zero-loss
    modifier.  Created because it is unclear what the deconvolution 
    methods in hyperspy are doing.

    Paramters
    ---------
    HL : 

    LL : 

    stray : Boolean
        If True, treat low-loss to remove stray signal in low-loss.
    pad : Boolean
        If True, pads the high-loss end of the spectra with a decay to zero
    plot : Boolean
        If true, plots are given of intermediate stages.

    Returns
    -------
    reconv : hyperspy spectrum object


    '''
    # Make copies of data to manipulate
    low = LL.copy()
    high = HL.copy()
    reconv = HL.deepcopy()
    # Record high-loss offset, as this is lost during convolution
    HL_offset = HL.axes_manager[0].offset
    HL_size = HL.axes_manager[0].size
    # Remove stray signal
    if stray:
        temp = LL.copy()
        LL = remove_stray_signal(temp, 0.8)
    # Pad spectra for size-matching and continuity at boundaries
    if pad:
        # TODO
        low, high = match_spectra_sizes(LL, HL)
    
    
    # Extract the Zero-loss Peak for the modifier
    ZLP = extract_ZLP(low, method='reflected tail', plot=plot)
    # Calculate Fourier Transforms.
    LLF = fft(low.data)
    HLF = fft(high.data)
    ZLF = fft(ZLP.data)
    # Compute Convolution
    conv = (HLF / LLF) * ZLF
    # Extract real part of the inverse transform to get convolved signal back
    iconv = np.real(ifft(conv))
    # Restore high-loss spectrum dimensions
    reconv.data = iconv[:HL_size]

    return reconv

def reverse_fourier_ratio_convoln(HL, LL, stray=False, pad=True, plot=False):
    '''
    Function for performing Fourier-Ratio deconvolution using a zero-loss
    modifier.  Created because it is unclear what the deconvolution 
    methods in hyperspy are doing.

    Paramters
    ---------
    HL :

    LL : 

    stray : Boolean
        If True, treat low-loss to remove stray signal in low-loss.
    pad : Boolean
        If True, pads the high-loss end of the spectra with a decay to zero
    plot : Boolean
        If true, plots are given of intermediate stages.

    Returns
    -------
    reconv : hyperspy spectrum object


    '''
    # Make copies of data to manipulate
    low = LL.copy()
    high = HL.copy()
    reconv = HL.deepcopy()
    # Record high-loss offset, as this is lost during convolution
    HL_offset = HL.axes_manager[0].offset
    HL_size = HL.axes_manager[0].size
    # Remove stray signal
    if stray:
        temp = LL.copy()
        LL = remove_stray_signal(temp, 0.8)
    # Pad spectra for size-matching and continuity at boundaries
    if pad:
        # TODO
        low, high = match_spectra_sizes(LL, HL)
    
    
    # Extract the Zero-loss Peak for the modifier
    ZLP = extract_ZLP(low, method='reflected tail', plot=plot)
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
