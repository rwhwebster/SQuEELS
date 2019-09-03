from __future__ import print_function

import numpy as np
import scipy as sp

import pymc3 as pm

import matplotlib.pyplot as plt
plt.ion()

def dummy_quant(a):
    return a
    
def create_bayes_model(stds, comps, data, data_range, guesses=1.0, plot=False):
    '''
    Initialised a pymc3 model using the active standards library

    Parameters
    ----------
    stds : Standards Library

    comps : list of strings

    data : Hyperspy Spectrum Object

    data_range : tuple of floats

    guesses : tuple of floats
        

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
        beta = []
        for i, comp in enumerate(comps):
            beta.append(pm.Normal(comp, mu=guesses[i], sigma=1))
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

def fit_bayes_model(model=None, nDraws=1000, params=None, plot=False):
    '''
    Run Monte-Carlo chains on model and return fit results.

    Parameters
    ----------
    model : pymc3 model
        A model prepared using create_bayes_model
    sample_params : dict
        contains arguments to be passed to the sampling function
        See pymc3.sample for arguments.
    plot : boolean
        If true, plot details of the final trace
    Returns
    -------

    '''

    if model is None:
        raise Exception("model=None method does not exist.")

    # Prepare parameters

    with model:
        trace = pm.sample(nDraws, **params)

    if plot:
        pm.traceplot(trace)

    return trace