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
    vals = generate_hann_window(diff)
    vals *= a0
    # Update data in out
    out.data = np.zeros((nLen))
    out.data[:oLen] = s.data
    out.data[oLen:int(oLen+diff/2)] = vals[:int(diff/2)]

    return out

def zero_pad_spectrum(s, nLen, axis):
    '''
    Pads the spectrum to match the length nLen given along axis and fills
    the expanded region with zeros.

    Parameters
    ----------
    s : hyperspy signal
        The spectrum to be padded
    nLen : int
        final length of padded spectrum along specified axis
    axis : int
        the axis along which to extend the spectrum

    Returns
    -------
    out : hyperspy signal
        the padded spectrum/SI
    '''
    out = s.deepcopy()
    oLen = s.axes_manager[axis].size
    out.axes_manager[axis].size = nLen
    dims = list(s.data.shape)
    dims[axis] = nLen
    out.data = np.zeros(dims)
    out.data[...,:oLen] = s.data
    return out


def match_spectra_sizes(s1, s2, taper=True):
    '''
    Engineer two spectra to have the same dimensions on the energy-loss axis.
    Achieves this by using pad_spectrum() to resize spectra to a value that
    satisfies 2^N.

    Parameters
    ----------
    s1 : Hyperspy signal
        One of the two spectra to be size matched.
    s2 : Hyperspy signal
        One of the two spectra to be size matched.

    Returns
    -------
    o1 : Hyperspy signal
        The size-matched version of input s1.
    o2 : Hyperspy signal
        The size-matched version of input s2.
    '''
    sigDim = s1.axes_manager.signal_indices_in_array[0]
    l1 = s1.axes_manager[sigDim].size
    l2 = s2.axes_manager[sigDim].size
    # l1 = len(s1.data)
    # l2 = len(s2.data)
    # Determine the smallest 2^N that satisfies the sizes of both inputs
    # then add 1 to N to ensure reasonably sized padding region
    n1 = int(np.ceil(np.log2(l1)))
    n2 = int(np.ceil(np.log2(l2)))
    N = max(n1,n2) + 1

    k = pow(2, N)

    o1 = zero_pad_spectrum(s1, k, sigDim)
    o2 = zero_pad_spectrum(s2, k, sigDim)

    return o1, o2

def generate_reflected_tail(s, centreChannel, threshold):
    '''
    Takes the input spectrum, s, and returns only the largest peak, replacing
    other data present using the reflected tail method.

    Parameters
    ----------
    s : hyperspy spectrum

    centreChannel : int

    threshold : float
        Fraction of max peak intensity which is used to define where tails
        are taken from.

    Returns
    -------
    out : hyperspy spectrum

    '''
    out = s.deepcopy()

    # Seek left of peak for threshold
    amp = s.data[centreChannel]
    current = amp
    i = centreChannel
    while (current/amp) > threshold:
        i -= 1
        current = s.data[i]
    left = i
    # now seek right
    current = amp
    i = centreChannel
    while (current/amp) > threshold:
        i += 1
        current = s.data[i]
    right = i
    # Extract tail and reflect, removing other data
    tail = s.data[:left]
    out.data[right:] = 0.0
    out.data[right:right+left] = tail[::-1]

    return out


def extract_ZLP(s, method='reflected tail', threshold=0.02, plot=False):
    '''
    Examines spectrum s for zero-loss peak and returns a dataset of the same
    dimensions, but which only contains the ZLP.

    Parameters
    ----------
    s : hyperspy spectrum object
        Must be a low-loss spectrum which contains the 0eV channel.
    method : string
        Procedure for extracting the ZLP.  Available options are:
        - 'fit' : Fits a skewed Voigt function to the ZLP.
        - 'reflected tail' : Creates a ZLP model by reflecting the energy
            gain side of the ZLP about the maximum.
    threshold : float
        For use with reflected tail method, determines cutoff for tail.
    plot : Boolean
        If true, plots the extracted ZLP over the low-loss data.

    Returns
    -------
    ZLP : hyperspy spectrum object
        spectrum identical to s, but data contains only the extracted zero-loss
        peak.

    
    TO DO: Further testing on ZLPs of different dispersions. Confirm SkewedVoigtModel
    is a reasonable choice, but could benefit from further optimisation.
    '''

    offset = s.axes_manager[0].offset
    scale = s.axes_manager[0].scale
    zlpChannel = -offset/scale
    if zlpChannel < 0 or zlpChannel > s.axes_manager[0].size:
        raise Exception('Spectrum provided does not contain the 0eV channel.')
    zlpChannel = np.argmax(s.data)

    methods = { 'reflected tail' : generate_reflected_tail(s, zlpChannel, threshold),
                'fit' : print("method needs updated.")
    }

    ZLP = methods.get(method)

    if plot:
            plt.figure()
            plt.plot(s.data, label='data')
            plt.plot(ZLP.data, label=method+' model')
            plt.show()

    return ZLP