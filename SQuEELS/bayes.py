from __future__ import print_function

import gc

import numpy as np

import pandas as pd

from tqdm import trange

import theano
import pymc3 as pm

import matplotlib.pyplot as plt
plt.ion()

class BayesModel_old:
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


class BayesModel:
    '''

    '''
    def __init__(self, core_loss, standards):
        '''

        '''
        self.stds = standards
        self.HL = core_loss

        self.dims = core_loss.data.data.shape
        self.nDims = len(self.dims)
        self.sigDim = core_loss.data.axes_manager.signal_indices_in_array[0]

    def model_at_index(self, coords, prior_means=None, bkgd_model='power law', width=100.0, nSamples=None,  init_params={}, nDraws=1000, chain_params={}):
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
        #yx = self._map_yx(nSamples)
        #nSamples = len(yx) # Number of different samples in dataset
        yx = coords
        # Second, set up dataframe to hold model results for each spectrum
        #df = pd.DataFrame(columns=['Y','X','Trace',*self.comps])

        # Third, prepare observed data and references as theano.shared instances
        # Observable data
        data = theano.shared(self.HL.data.inav[*yx].data) # Use first point in coordinate list 'yx'
        # Reference spectra
        self.stds.model = dict()
        for ref in self.comps:
            self.stds.model[ref] = theano.shared(self.stds.ready[ref].inav[*yx].data)

        x = self.HL.data.axes_manager[-1].axis

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
            if bkgd_model=='power law':
                mu += beta[-2]*x

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