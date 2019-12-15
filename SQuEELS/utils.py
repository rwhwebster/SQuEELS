from __future__ import print_function

import numpy as np
import scipy as sp

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

