from __future__ import print_function

import numpy as np
import scipy as sp

import pymc3 as pm

import matplotlib.pyplot as plt

from .processing import remove_stray_signal
plt.ion()

class BayesModel:
    '''
    Class for handling Bayesian analysis of elemental composition.
    '''
    def __init__(self, core_loss, stds, comps, data_range, low_loss=None):
        '''
        Parameters
        ----------
        core_loss : Hyperspy spectrum (image)

        stds : SQuEELS Standards object

        comps : list of strings

        data_range : length 2 tuple of floats
        
        low_loss : Hyperspy spectrum (image)

        '''

        self.stds = stds
        self.stds.set_all_inactive()
        
        self.comps = comps
        self.stds.set_active_standards(self.comps)

        if low_loss:
            self.LL = low_loss

        self.dims = core_loss.data.shape
        self.nDims = len(self.dims)
        sigDim = core_loss.axes_manager.signal_indices_in_array[0]

        try:
            self.HL = core_loss.deepcopy()
            self.HL.crop(start=data_range[0], end=data_range[1], axis=sigDim)
        except:
            raise Exception('Something went wrong cropping core-loss data.')

        self.stds.set_spectrum_range(*data_range)



    def init_model(self, nav=None, mu_0=None, deStray=None, strayArgs={'method':1}, zlpArgs={}):
        '''
        Initialise a model for the current spectrum.

        Parameters
        ----------
        nav : length 2 list of ints
            The inav coordinates of the spectrum within the SI to be
            quantified. Do not provide if model was initialised with
            single spectrum.
        mu_0 : len(comps) tuple of floats
            Initial guesses to help model reach solution faster.
            Optional.
        '''
        # Remove stray signal if requested.
        if deStray:
            temp = self.LL.deepcopy()
            self.LL = remove_stray_signal(temp, deStray.pop('method'), *strayArgs)
        # Forward convolve references if LL is provided
        if self.LL:
            if nav is None:
                currentL = self.LL
            else:
                currentL = self.LL.inav[nav]
            self.stds.convolve_ready(currentL, kwargs=zlpArgs)
            self.stds.model = self.stds.conv
        else:
            self.stds.model = self.stds.ready
        # Prepare relevant core-loss data
        if nav is None:
            self._Y = self.HL
        else:
            self._Y = self.HL.inav[nav]

        if not mu_0:
            mu_0 = (1.0,)*len(self.comps)
        # Create pm model instance
        with pm.Model() as self.model:
            beta = []
            for i, comp in enumerate(self.comps):
                beta.append(pm.Normal(comp, mu=mu_0[i], sigma=1))
            sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)

            mu = None

            for i, comp in enumerate(self.comps):
                if mu is None:
                    mu = beta[i]*self.stds.model[comp].data
                else:
                    mu += beta[i]*self.stds.model[comp].data

            likelihood = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=self._Y.data)

    def start_chains(self, nDraws=1000, params=None, plot=False):
        '''
        
        '''
        if self.model is None:
            raise Exception("The step 'init_model' needs to be run first.")


        with self.model:
            self.trace = pm.sample(nDraws, **params)

        if plot:
            self.show_results()

    def show_results(self):
        '''
        
        '''
        if self.trace:
            # Plot trace histograms
            pm.traceplot(self.trace)
            stats = pm.summary(self.trace)
            print(stats)
            #Plot fit over data
            xvals = self._Y.axes_manager[0].axis
            fig, ax = plt.subplots(1,1)
            ax.plot(xvals, self._Y.data)
            fit = self._Y.data.copy()*0
            for comp in self.comps:
                par = self.stds.model[comp].data*stats['mean'][comp]
                fit += par
                ax.plot(xvals, par)
            ax.plot(xvals, fit)
            ax.legend(['Data',*self.comps,'Fit'])
            ax.set_xlabel('Energy-loss (eV)')
            fig.suptitle("Comparison of fit and data")
            fig.show()
            from pandas.plotting import scatter_matrix
            scatter_matrix(pm.trace_to_dataframe(self.trace), figsize=(10,10))
        else:
            raise Exception('No trace has been computed to visualise.')

# END OF CLASS




def create_bayes_model(stds, comps, data, data_range, guesses=(1.0,), LL=None, plot=False):
    '''
    Initialised a pymc3 model using the active standards library

    Parameters
    ----------
    stds : Standards Library

    comps : list of strings

    data : Hyperspy Spectrum Object

    data_range : tuple of floats

    guesses : tuple of floats
        
    LL : Hyperspy spectrum object

    plot : boolean

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
    # If Low-loss is provided, convolve standards
    if LL:
        stds.convolve_ready(LL, kwargs={'stray':True})
        stds.model = stds.conv
    else:
        stds.model = stds.ready
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
                mu = beta[i]*stds.model[comp].data
            else:
                mu += beta[i]*stds.model[comp].data

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
