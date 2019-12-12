from __future__ import print_function

import gc

import numpy as np

import pandas as pd

from tqdm import trange

import theano
import pymc3 as pm

import matplotlib.pyplot as plt

from .processing import remove_stray_signal
plt.ion()

class BayesModel:
    '''
    Class for handling Bayesian analysis of elemental composition.
    '''
    def __init__(self, core_loss, stds, comps, data_range, low_loss=None,
        normalise=True, lognorm=False):
        '''
        Parameters
        ----------
        core_loss : SQuEELS Data object
            The core-loss data to be modelled.  Can be single spectrum or
            spectrum image.
        stds : SQuEELS Standards object
            The standards library object loaded by the class method in 
            SQuEELS.io
        comps : list of strings
            The names of the references in the standards library which are
            to be used as components in the model.
        data_range : length 2 tuple of floats
            The start and end energies over which to model. Must be provided
            as floats, ints will do weird stuff because of the way Hysperspy
            operates.
        low_loss : Hyperspy spectrum (image)
            If provided, the low-loss spectrum is used to forward-convolve the
            components standards before modelling.
        normalise : boolean
            If True, normalises intensities of data and references.
        lognorm : boolean
            If True, takes the natural log of data and references before
            normalisation.
        '''

        self.stds = stds
        self.stds.set_all_inactive()
        
        self.comps = comps
        self.stds.set_active_standards(self.comps)

        if low_loss:
            self.LL = low_loss
        else:
            self.LL = None

        self.dims = core_loss.data.data.shape
        self.nDims = len(self.dims)
        sigDim = core_loss.data.axes_manager.signal_indices_in_array[0]

        try:
            self.HL = core_loss
            self.HL.data.crop(start=data_range[0], end=data_range[1], axis=sigDim)
        except:
            raise Exception('Something went wrong cropping core-loss data.')

        self.stds.set_spectrum_range(*data_range)

        if normalise:
            self.stds.normalise(logscale=lognorm)
            if lognorm:
                self.HL.apply_logscale()
            self.HL.normalise()



    # def init_model(self, nav=None, mu_0=None, strayArgs=None, strayKwargs={}, zlpArgs={}):
    #     '''
    #     Initialise a model for the current spectrum.  This is separate from the 
    #     __init__ method of the class, so that this can be repeated for multiple
    #     spectra in an SI.

    #     Parameters
    #     ----------
    #     nav : length 2 list of ints
    #         The inav coordinates of the spectrum within the SI to be
    #         quantified. Do not provide if model was initialised with
    #         single spectrum.
    #     mu_0 : len(comps) tuple of floats
    #         Initial guesses to help model reach solution faster.
    #         Optional.
    #     deStray : Boolean
    #         If true, removes a stray signal shape from the low-loss.
    #     strayArgs : dict
    #         Arguments to be provided to the remove_stray_signal call.
    #         See SQuEELS.processing.remove_stray_signal() for more info.
    #     zlpArgs : dict
    #         Specify any arguments you wish to feed to the extract_ZLP call
    #         to how it is used in the forward convolution of standards.
    #     '''
    #     # Remove stray signal if requested.
            
    #     # Forward convolve references if LL is provided
    #     if self.LL:
    #         if nav is None:
    #             currentL = self.LL
    #         else:
    #             currentL = self.LL.inav[nav]
    #         if strayArgs is not None:
    #             temp = currentL.deepcopy()
    #             currentL = remove_stray_signal(temp, *strayArgs, **strayKwargs)
    #         self.stds.convolve_ready(currentL, kwargs=zlpArgs)
    #         self.stds.model = self.stds.conv
    #     else:
    #         self.stds.model = self.stds.ready
    #     # Prepare relevant core-loss data
    #     if nav is None:
    #         self._Y = self.HL
    #     else:
    #         self._Y = self.HL.inav[nav]

    #     if not mu_0:
    #         mu_0 = (1.0,)*len(self.comps)
    #     # Create pm model instance
    #     with pm.Model() as model:
    #         beta = []
    #         for i, comp in enumerate(self.comps):
    #             beta.append(pm.Normal(comp, mu=mu_0[i], sigma=1))
    #         sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)

    #         mu = None

    #         for i, comp in enumerate(self.comps):
    #             if mu is None:
    #                 mu = beta[i]*self.stds.model[comp].data
    #             else:
    #                 mu += beta[i]*self.stds.model[comp].data

    #         likelihood = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=self._Y.data)

    #     return model

    # def start_chains(self, model, nDraws=1000, params=None, plot=False):
    #     '''
    #     Once the model is initialised using BayesModel.init_model(), start
    #     the Monte Carlo chains to perform the modelling and get results.

    #     Parameters
    #     ----------
    #     nDraws : int
    #         Number of Monte Carlo steps to be carried out in each chain.
    #     params : dict
    #         Keyword arguments to be passed to pm.sample().  See pymc3 docs
    #         for more details.
    #     plot : boolean
    #         If true, makes a call to BayesModel.show_results() to give a
    #         visual summary of the model results.
    #     '''
    #     if model is None:
    #         raise Exception("The step 'init_model' needs to be run first.")


    #     with model:
    #         trace = pm.sample(nDraws, **params)

    #     if plot:
    #         self.show_results()

    #     return trace

    # def show_results(self, trace=None, nav=[0,0]):
    #     '''
    #     Visualise the results of the most recent BayesModel.start_chains()
    #     '''
    #     if not trace:
    #         try:
    #             trace=self.trace
    #             _Y = self._Y
    #         except:
    #             raise Exception('No trace provided to visualise.')
    #     else:
    #         _Y = self.HL.inav[nav]
    #     # Plot trace histograms
    #     pm.traceplot(trace)
    #     stats = pm.summary(trace)
    #     print(stats)
    #     #Plot fit over data
    #     xvals = _Y.axes_manager[0].axis
    #     fig, ax = plt.subplots(1,1)
    #     ax.plot(xvals, _Y.data)
    #     fit = _Y.data.copy()*0
    #     for comp in self.comps:
    #         par = self.stds.model[comp].data*stats['mean'][comp]
    #         fit += par
    #         ax.plot(xvals, par)
    #     ax.plot(xvals, fit)
    #     ax.legend(['Data',*self.comps,'Fit'])
    #     ax.set_xlabel('Energy-loss (eV)')
    #     fig.suptitle("Comparison of fit and data")
    #     fig.show()
    #     from pandas.plotting import scatter_matrix
    #     scatter_matrix(pm.trace_to_dataframe(trace), figsize=(10,10))

    def _random_sample(self, nSpectra):
        '''
        Create mapping for random sampling of spectra from a spectrum image.

        Note: current implementation allows scope for duplicate entries,
            (not desirable)

        Parameters
        ----------
        nSpectra : int
            The number of spectra to sample from the SI.

        Returns
        -------
        yx_map : numpy.ndarray
            nSpectra x 2 list of array coordinates.
        '''

        if self.nDims < 2:
            raise Exception('BayesModel._random_sample requires signal data with > 1 dimensions.')
        # Create randomised list of coordinates  which lie within spectrum image
        coords = []
        for axis in range(self.nDims-1):
            coords.append(np.random.randint(0, high=self.dims[axis], size=nSpectra))
        temp = np.array(coords)
        yx_map = temp.T

        return yx_map

    def _full_sample(self):
        '''
        Create mapping for complete sampling of spectra from a spectrum image.

        Parameters
        ----------
        nSpectra : int
            The number of spectra to sample from the SI.

        Returns
        -------
        yx_map : numpy.ndarray
            nSpectra x 2 list of array coordinates.
        '''

        if self.nDims < 2:
            raise Exception('BayesModel._random_sample requires signal data with > 1 dimensions.')
        # Create randomised list of coordinates  which lie within spectrum image
        temp = []
        for y in range(self.dims[0]):
            for x in range(self.dims[1]):
                temp.append([y,x])
        yx_map = np.array(temp)

        return yx_map

    def _map_yx(self, nSamples):
        '''
        Method which decides how to construct list of array coordinates based
        on the type of data provded to nSamples.

        Parameters
        ----------
        nSamples : None, int or ndarray

        Returns
        -------
        yx : ndarray

        '''
        if nSamples is not None:
            ntype = type(nSamples)
            if ntype is int: # If int, get random selection of array coordinates
                yx = self._random_sample(nSamples, ret=True)
            elif ntype is np.ndarray: # if ndarray, no more needs done
                yx = nSamples
            else:
                raise Exception('No method defined for "nSamples" input type.')
        else:
            # Else, generate list of coordinates that covers full array
            yx = self._full_sample()

        return yx

    def simple_multimodel(self, width=1.0, nSamples=None, prior_means=None, init_params={}, nDraws=1000, chain_params={}, retain_trace=False):
        '''
        Get Bayesian statistics for multiple spectra in the spectrum image.
        Provides no scope for forward convolution of references.

        Parameters
        ----------
        nSamples : None, int or ndarray
            Leave as None, or specify a number of random samples to draw
            from the SI.
        prior_means : None or tuple of floats
            Initial guesses for each of the components to be fed to the model
        init_params : dict
            Dictionary of arguments and kwargs taken by init_model.
        nDraws : int
            The number of monte carlo iterations each chain should run.
            The number of chains can be specified in chain_params.
        chain_params : dict
            Dictionary of arguments and kwargs taken by start_chains.
        retain_trace : bool
            If true, the raw trace data is included in the dataframe.
            WARNING: This method consumes a lot of memory.

        Returns
        -------
        df : pandas dataframe
            Dataframe containing the output of the monte carlo chains.
        '''
        if prior_means is None:
            mu_0 = (1.0,)*len(self.comps)
        else:
            mu_0 = prior_means
        # First, establish list of array coordinates to operate over
        yx = self._map_yx(nSamples)
        nSamples = len(yx) # Number of different samples in dataset

        # Second, set up dataframe to hold model results for each spectrum
        df = pd.DataFrame(columns=['Y','X','Trace',*self.comps])

        # Third, prepare observed data and references as theano.shared instances
        # Observable data
        data = theano.shared(self.HL.data.inav[yx[0,1], yx[0,0]].data) # Use first point in coordinate list 'yx'
        # Reference spectra
        self.stds.model = dict()
        for ref in self.comps:
            self.stds.model[ref] = theano.shared(self.stds.ready[ref].data)

        # Fourth, create the bare-bones of a pymc3 model
        with pm.Model() as model:
            beta = []
            for i, comp in enumerate(self.comps):
                beta.append(pm.Normal(comp, mu=mu_0[i], sigma=width))
            sigma = pm.HalfCauchy('sigma', beta=10, testval=width)

            mu = None

            for i, comp in enumerate(self.comps):
                if mu is None:
                    mu = beta[i]*self.stds.model[comp]
                else:
                    mu += beta[i]*self.stds.model[comp]

            pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=data)
        # End of model declaration

        # Final stage is to run loop over spectra to calculate traces        
        for i in trange(nSamples, desc='Sampling spectra from SI', unit='spectra'):
            y = yx[i,1]
            x = yx[i,0]
            # Update shared data to current spectrum
            data.set_value(self.HL.data.inav[y,x].data)
            # Prepare results dict
            results = {'Y':y, 'X':x}
            try:
                with model:
                    results['Trace'] = pm.sample(nDraws, **chain_params)
                smry = pm.summary(results['Trace'])
                for comp in self.comps:
                   results[comp] = np.mean(results['Trace'].get_values(comp))
                   results[comp+' sd'] = smry['sd'][comp]
                   results[comp+' hpd_97.5'] = smry['hpd_97.5'][comp]
                   results[comp+' hpd_2.5'] = smry['hpd_2.5'][comp]
            except:
                print('Model at index '+str(i)+' has failed.')
                results['Trace'] = np.nan
                for comp in self.comps:
                   results[comp] = np.nan
                   results[comp+' sd'] = np.nan
                   results[comp+' hpd_97.5'] = np.nan
                   results[comp+' hpd_2.5'] = np.nan
            # Append current results to dataframe
            df = df.append(results, ignore_index=True)

        return df


# END OF CLASS
