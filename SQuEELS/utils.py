from __future__ import print_function

import numpy as np
import scipy as sp

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()


def generate_hann_window(dims, direction=0):
    '''
    Parameters
    ----------
    dims : int or tuple of ints
        dimensions of the ndarray that the Hanning window will be created in.
    direction : int
        Determines axis along which window varies, does not need to be used for
        1d arrays.

    Returns
    -------
    window : ndarray
        array containing the hanning window
    '''
    window = np.zeros(dims)
    # This is a teeny-tiny amount of future-proofing
    if isinstance(dims, int):
        dims = (dims,)

    window_swapped = np.swapaxes(window, 0, direction)

    for i in range(dims[direction]):
        window_swapped[i] = np.cos(2*np.pi*i/dims[direction]) # This is only valid for 'right'

    window = np.swapaxes(window_swapped, 0, direction)
    
    window = 0.5 * window + 0.5

    return window

def pad_spectrum(s, nLen):
    '''
    Pads the high energy-loss end of the spectrum to match the given length
    and fills the new data by given method.

    Parameters
    ----------
    s : 
        Spectrum to be expanded
    nLen : int
        Length of spectrum once padded

    Returns
    -------
    out : 
        The extended spectrum
    '''
    out = s.deepcopy()
    oLen = s.axes_manager[0].size
    out.axes_manager[0].size = nLen
    diff = nLen - oLen
    # Determine amplitide of high-loss tail of spectrum
    a0 = np.mean(s.data[-20:])
    # Create data to fill pad.
    # TODO Generalise to make other methods implementable
    vals = make_hann_window(diff)
    vals *= a0
    # Update data in out
    out.data = np.zeros((nLen))
    out.data[:oLen] = s.data
    out.data[oLen:oLen+diff/2] = vals[:diff/2]

    return out

def match_spectra_sizes(s1, s2):
    '''
    Engineer two spectra to have the same dimensions on the energy-loss axis.
    Achieves this by using pad_spectrum() to resize spectra to a value that
    satisfies 2^N.

    Parameters
    ----------
    s1 : Hyperspy spectrum
        One of the two spectra to be size matched.
    s2 : Hyperspy spectrum
        One of the two spectra to be size matched.

    Returns
    -------
    o1 : Hyperspy spectrum
        The size-matched version of input s1.
    o2 : Hyperspy spectrum
        The size-matched version of input s2.
    '''
    l1 = len(s1.data)
    l2 = len(s2.data)
    # Determine the smallest 2^N that satisfies the sizes of both inputs
    # then add 1 to N to ensure reasonably sized padding region
    n1 = int(np.ceil(np.log2(l1)))
    n2 = int(np.ceil(np.log2(l2)))
    N = max(n1,n2) + 1

    k = pow(2, N)

    o1 = pad_spectrum(s1, k)
    o2 = pad_spectrum(s2, k)

    return o1, o2