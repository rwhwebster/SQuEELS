from __future__ import print_function

import numpy as np
import scipy as sp

# import numba
# from numba import jit

import matplotlib.pyplot as plt
plt.ion()

# @jit(nopython=True)
def _normalEqn(X, y):
    '''
    Calculate solution to linear regression parameters by matrix inversion.

    Parameters
    ----------
    X : ndarray
        Matrix of fit components
    y : ndarray
        Observed data points
    Returns
    -------
    theta : ndarray
        Optimised model parameters
    '''
    theta = np.zeros((X.shape[1], 1))
    
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

    return theta

def solo_normal_solver(stds, comps, data, data_range, LL=None, plot=False):
    '''
    Solve multivariate linear regression using matrix inversion.

    Parameters
    ----------
    stds : 

    comps : 

    data : 

    data_range : 

    LL : 

    plot : 
    Returns
    -------
    theta : 
    '''
    # First, reset standards library
    stds.set_all_inactive()
    # Then use comps to specify which Standards are to be used
    stds.set_active_standards(comps)
    # Now, use the data range provided to set the fit range in the data
    try:
        Y = data.deepcopy()
        Y.crop(start=data_range[0], end=data_range[1], axis=0)
    except:
        raise Exception('Problem cropping observed data.')
    # If crop of data succeeds, apply same to active Standards
    stds.set_spectrum_range(data_range[0], data_range[1])
    # If Low-loss is provided, convolve standards
    if LL:
        stds.convolve_ready(LL, kwargs={'stray':True})
        stds.model = stds.conv
    else:
        stds.model = stds.ready

    # Build component matrix
    x_mat = np.array([stds.model[comp].data for comp in comps]).T
    y_obs = Y.data.T
    # Now compute solution
    theta = _normalEqn(x_mat, y_obs)

    return theta
