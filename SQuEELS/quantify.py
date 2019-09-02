from __future__ import print_function

import numpy as np
import scipy as sp

import pymc3 as pm

import matplotlib.pyplot as plt
plt.ion()

def dummy_quant(a):
    return a
    
def create_bayes_model(stds, comps, data, data_range, guess=1.0, plot=False):
    '''
    Initialised a pymc3 model using the active standards library

    Parameters
    ----------
    stds : Standards Library

    comps : list of strings

    data : Hyperspy Spectrum Object

    data_range : tuple of floats
        

    Returns
    -------
    model : pymc3 model

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
    # Create model
    with pm.Model() as model:
        beta = pm.Normal('beta', mu=guess, sigma=1, shape=len(comps))
        # sigma = pm.HalfNormal('sigma', sigma=1)
        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)

        mu = None

        for i, comp in enumerate(comps):
            if mu is None:
                mu = beta[i]*stds.ready[comp].data
            else:
                mu += beta[i]*stds.ready[comp].data

        likelihood = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y.data)

    if plot:
        plt.figure()
        plt.plot(Y.data)
        for comp in comps:
            plt.plot(stds.ready[comp].data)

    return model

