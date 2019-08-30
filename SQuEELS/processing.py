from __future__ import print_function

import numpy as np
import scipy as sp

from lmfit import models

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()

def dummy_proc(a):
    return a
    
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

    if isinstance(dims, int):
        dims = (dims,)

    for i in range(dims[direction]):
        window[i] = np.cos(np.pi*i/dims[direction] + np.pi) # This is only valid for 'right'

    window = 0.5 - 0.5 * window

    return window

def extract_ZLP(s, plot=False):
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
    '''

    # Spectrum must contain zero-channel
    offset = s.axes_manager[0].offset
    scale = s.axes_manager[0].scale
    zlpChannel = -offset/scale
    if zlpChannel < 0 or zlpChannel > s.axes_manager[0].size:
        raise Exception('Spectrum provided does not contain the 0eV channel.')

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

    if plot:
        plt.figure()
        plt.plot(x, y, label='data')
        plt.plot(x, fitted, label='fit')
        plt.show()

    ZLP = s.deepcopy()
    ZLP.data = fitted

    return ZLP