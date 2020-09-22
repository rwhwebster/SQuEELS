from __future__ import print_function

import numpy as np
import scipy as sp

from lmfit import models

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()


def remove_stray_signal(s, sig_range, method, stray_shape=None, smooth=True):
    '''
    Method for identifying and removing stray signal under the low-loss
    spectrum.  Stray signal manifests as intensity before the zero-loss peak.

    Parameters
    ----------
    s : Hyperspy spectrum
        The spectrum from which to remove signal.
    sig_range : tuple of floats/ints
        The start and end energies of the window over which to integrate
        signal to determine scaling factor for stray shape.
    method : int
        decides which procedure to use for removing stray signal. Available
        options are:
            - 0 : Flat subtraction of the mean value within sig_range from the
                whole spectrum.
            - 1 : Provide a .dm3 file with a stray shape to subtract. The shape
            is scaled using the ratio of the intensities in sig_range
    stray_shape : string
        Path to the file containing the stray shape to be used for method 1.
        If None, opens a file browser.
    smooth : Boolean
        If true, nulls all the data to the left of the zero-loss from the point
        closest to the ZLP where negative intensity is encountered.  
        This part of the method needs refined.

    Returns
    -------
    out : Hyperspy spectrum
        The stray corrected low-loss spectrum.
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
        if not stray_shape:
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
