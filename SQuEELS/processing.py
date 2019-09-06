from __future__ import print_function

import numpy as np
import scipy as sp

from lmfit import models

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()

    
def make_hann_window(dims, side, direction=0):
    '''
    Parameters
    ----------
    dims : int or tuple of ints
        dimensions of the ndarray that the Hanning window will be created in.
    side : str
        Can be 'left', 'right', or 'both'.  Determines which part of the hann
        window to create.
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

    for i in range(dims[direction]):
        window[i] = np.cos(np.pi*i/dims[direction] + np.pi) # This is only valid for 'right'

    window = 0.5 - 0.5 * window

    return window

def extract_ZLP(s, method='fit', plot=False):
    '''
    Examines spectrum s for zero-loss peak and returns a dataset of the same
    dimensions, but which only contains the ZLP.

    Parameters
    ----------
    s : hyperspy spectrum object
        Must be a low-loss spectrum which contains the 0eV channel.
    plot : Boolean
        Determines whether to plot results or not.

    Returns
    -------
    ZLP : hyperspy spectrum object
        spectrum identical to s, but data contains only the extracted zero-loss
        peak.

    TODO
    ----
    Further testing on ZLPs of different dispersions. Confirm SkewedVoigtModel
    is a good choice.
    '''

    # Spectrum must contain zero-channel

    offset = s.axes_manager[0].offset
    scale = s.axes_manager[0].scale
    zlpChannel = -offset/scale
    if zlpChannel < 0 or zlpChannel > s.axes_manager[0].size:
        raise Exception('Spectrum provided does not contain the 0eV channel.')
    zlpChannel = np.argmax(s.data)

    if method=='fit':
        # Create Model for ZLP
        mod = models.SkewedVoigtModel(prefix='ZLP')
        mod.set_param_hint('ZLPcenter', value=0.0, min=-1.0, max=1.0)
        mod.set_param_hint('ZLPamplitude', value=max(s.data))
        mod.set_param_hint('ZLPsigma', value=1/scale)
        mod.set_param_hint('ZLPgamma', value=10)

        # prepare data for fit
        buf = 10
        x = s.axes_manager[0].axis
        y = s.data
        weights = y.copy()*0
        weights[int(zlpChannel-buf/scale):int(zlpChannel+buf/scale)] = 1.0

        result = mod.fit(y, x=x, method='leastsq', weights=weights)
        fitted = result.best_fit

        ZLP = s.deepcopy()
        ZLP.data = fitted

    elif method=='reflected tail':
        # crude method that assumes symmetric ZLP
        ZLP = s.deepcopy()
        reflect = int(np.round(zlpChannel)) + 1
        tail = s.data[:reflect]
        ZLP.data[reflect:] *= 0
        ZLP.data[reflect:2*reflect-1] = tail[::-1][1:]

    if plot:
            plt.figure()
            plt.plot(s.data, label='data')
            plt.plot(ZLP.data, label='fit')
            plt.show()

    return ZLP

def pad_spectrum(s, nLen, method='hann'):
    '''
    Pads the high energy-loss end of the spectrum to match the given length
    and fills the new data by given method.

    Parameters
    ----------
    s : 
        Spectrum to be expanded
    nLen : int
        Length of spectrum once padded
    method : str
        Currently, 'hann' is only available method.

    Returns
    -------
    out : 
        The extended spectrum
    '''
    out = s.deepcopy()
    oLen = s.axes_manager[0].size
    out.axes_manager[0].size = nLen
    # Determine amplitide of high-loss tail of spectrum
    a0 = np.mean(s.data[-20:])
    # Create data to fill pad.
    # TODO Generalise to make other methods implementable
    vals = make_hann_window(nLen-oLen, 'right')
    vals *= a0
    # Update data in out
    out.data = np.zeros((nLen))
    out.data[:oLen] = s.data
    out.data[oLen:] = vals

    return out

def match_spectra_sizes(s1, s2):
    '''
    Engineer two spectra to have the same dimensions on the energy-loss axis.
    Achieves this by using pad_spectrum() to resize spectra to a value that
    satisfies 2^N.

    Parameters
    ----------
    s1 : Hyperspy spectrum

    s2 : Hyperspy spectrum

    Returns
    -------
    o1, o2 : 

    '''
    l1 = len(s1.data)
    l2 = len(s2.data)
    # Determine the smallest 2^N that satisfies the sizes of both inputs
    n1 = int(np.ceil(np.log2(l1)))
    n2 = int(np.ceil(np.log2(l2)))
    N = max(n1,n2) + 2

    k = pow(2, N)

    o1 = pad_spectrum(s1, k)
    o2 = pad_spectrum(s2, k)

    return o1, o2

def remove_stray_signal(s, sig_range, method, stray_shape='browse', smooth=True):
    '''
    Method for identifying and removing stray signal under the low-loss
    spectrum.  Stray signal manifests as intensity before the zero-loss peak.

    Parameters
    ----------
    s : Hyperspy spectrum

    sig_range : tuple

    stray_shape : string

    method : int

    Returns
    -------
    out : Hyperspy spectrum

    '''
    # Get scale calibration details
    xo = s.axes_manager[0].offset
    xs = s.axes_manager[0].scale
    chan_range = np.divide(np.subtract(sig_range, xo), xs)
    # Dip into one of the methods
    if method==0:
        # Create a rough method first of all
        # 
        offset = np.mean(s.data[int(chan_range[0]):int(chan_range[1])])

        out = s - offset

        out.data[0:int(chan_range[1])] = 0.0

    elif method==1:
        # Load file containing stray signal
        if stray_shape=='browse':
            from tkinter import filedialog
            import tkinter as tk
            root=tk.Tk()
            root.withdraw()
            fp = filedialog.askopenfilename(filetypes=(('DM Files', '*.dm3'),))
            print('Retrieving stray shape from '+fp)
            stray = hs.load(fp).data
        else:
            stray = hs.load(stray_shape).data
        # Recast sig_range as channel numbers
        # Compare stray signal to spectrum to be corrected
        sig_sum = s.data[int(chan_range[0]):int(chan_range[1])].sum()
        stray_sum = stray[int(chan_range[0]):int(chan_range[1])].sum()
        mult = sig_sum/stray_sum
        stray *= mult
        out = s.deepcopy()
        out -= stray[:len(out.data)]

    else:
        raise Exception('Invalid method identifier specified.')

    if smooth:
        cont = True
        index = np.argmax(out.data)
        while cont:
            if out.data[index] < 0.0:
                out.data[:index+1] = 0.0
                cont = False
            else:
                index -= 1

    return out

def remedy_quadrant_glitch(s, gc=1024, width=10, plot=False):
    '''
    If there is a gain glitch due to the CCD quadrants, measure and correct.

    Parameters
    ----------
    s : ndarray
        The spectrum data to be corrected.
    gc : int
        Channel position of step glitch. For raw spectra from GIF Ultrascan
        camera, this occurs at the midpoint of the spectrum in channel 1024.
    width : int
        Number of channels to use either side of gc
    plot : bool
        If true, plot before and after results

    Returns
    -------
    out : ndarray
        The glitch-corrected spectral data.
    '''
    # Extract signal windows to average
    window_L = s[gc-width:gc]
    window_R = s[gc:gc+width]
    # Calculate Means
    mu_L = np.mean(window_L)
    mu_R = np.mean(window_R)
    # Difference between means
    delta = (mu_R - mu_L)/2.0
    # Make copy of spectrum to correct
    out = s.copy()
    # Make half-and-half correction
    out[:gc] += delta
    out[gc:] -= delta
    # Now plot
    if plot:
        plt.figure()
        plt.plot(s)
        plt.plot(out)

    return out
