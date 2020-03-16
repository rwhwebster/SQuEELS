from __future__ import print_function

from functools import partial
import multiprocessing as mp

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit as lsq

import pandas as pd

from tqdm import tqdm

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()


def _fit_model(data, bkgd_shape=None, lsqKwargs=None):
    '''
    Function that performs curve fitting based on use of
    scipy.optimize.curve_fit

    Params
    ------
    data : ndarray
        The data to be used in the model.  Prepared by the function
        MLLSmodel._prepare_model_data()
    bkgd_shape : string or None
        If not None, must be a string containing the name of one of
        the available background shapes:
        - power law
        - log-linear
        - double power law
    lsqKwargs : dict
        Keword arguments to be taken by scipy.optimize.curve_fit

    Returns
    -------
    opt_cofs : ndarray
        The optimised fit coefficients
    cov : ndarray
        The covariance matrix of the fit
    '''
    if lsqKwargs is None:
        lsqKwargs = {}

    y_obs = data[0,:]
    x = data[1,:]
    components = data[2:,:]

    def background(t, coeffs):
        if bkgd_shape=='power law':
            return coeffs[-2] * pow(t, coeffs[-1])
        if bkgd_shape=='log-linear':
            return np.log(coeffs[-2] * pow(t, coeffs[-1]))
        if bkgd_shape=='double power law':
            return coeffs[-4] * pow(t, coeffs[-3]) + coeffs[-2] * pow(t, coeffs[-1])

    def model(t, *coeffs):
        y = 0.0
        for i in range(components.shape[0]):
            y += coeffs[i] * components[i,:]
        if bkgd_shape:
            y += background(t, coeffs)
        return y

    opt_cofs, cov = lsq(model, x, y_obs, **lsqKwargs)

    # self.last = [opt_cofs, cov]
    return opt_cofs, cov


def _normalEqn(X, y):
    '''
    Calculate solution to linear regression parameters by matrix inversion.
    theta = ((X'X)^-1)X'y

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
    theta = np.zeros((X.shape[1], 1), dtype=np.float32)
    
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T).astype(np.float32), y.astype(np.float32))

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


class MLLSmodel_old:
    '''
    Class for handling standard MLLS fitting of spectra.
    '''
    def __init__(self, core_loss, standards):
        '''
        Parameters
        ----------
        core_loss : SQuEELS Data object
            The core-loss data to be modelled.
        standards : SQuEELS Standards object
            The standards library containing the references to be used
            as fit components
        '''

        self.stds = standards
        self.HL = core_loss

        self.dims = core_loss.data.data.shape
        self.nDims = len(self.dims)
        self.sigDim = core_loss.data.axes_manager.signal_indices_in_array[0]

    def model_point(self, coords, comps, init_guess, fit_background=None, lsqKwargs=None):
        '''
        
        Parameters
        ----------
        coords : tuple of ints

        comps : list of strings

        init_guess : list of floats

        fit_background : string or None


        Returns
        -------
        coefficients : array of floats

        '''
        if lsqKwargs is None:
            lsqKwargs = {}

        def background(t, coeffs):
            if fit_background=='power law':
                return coeffs[-2] * pow(t, coeffs[-1])
            if fit_background=='log-linear':
                return np.log(coeffs[-2] * pow(t, coeffs[-1]))
            if fit_background=='double power law':
                return coeffs[-4] * pow(t, coeffs[-3]) + coeffs[-2] * pow(t, coeffs[-1])

        def model(t, *coeffs):
            y = 0.0
            for i, comp in enumerate(comps):
                y += coeffs[i] * self.stds.ready[comp].inav[coords].data
            if fit_background:
                y += background(t, coeffs)
            return y

        y_Obs = self.HL.data.inav[coords].data
        t = self.HL.data.axes_manager[self.sigDim].axis

        opt_cofs, cov = lsq(model, t, y_Obs, p0=init_guess, **lsqKwargs)

        self.last = opt_cofs
        return opt_cofs

    def multimodel(self, comps, initial_guesses, background=None, update_guesses=True, lsqKwargs=None):
        '''
        This function is a wraparound for MLLSmodel.model_point which is used
        to process all spectra in a spectrum image.
        '''
        self.comps = comps
        df = pd.DataFrame(columns=['coord',*comps])

        points = np.prod(self.HL.data.axes_manager.shape[:-1])

        with tqdm(desc='Quantifying SI', total=points, unit='spectra') as pbar:
            for idx in np.ndindex(self.HL.data.axes_manager.shape[:-1]):
                results = {'coord': idx, }

                coeffs = self.model_point(idx, comps, initial_guesses, fit_background=background, lsqKwargs=lsqKwargs)

                for i, comp in enumerate(comps):
                    results[comp] = coeffs[i]
                if background:
                    results['bkgd_A'] = coeffs[-2]
                    results['bkgd_r'] = coeffs[-1]
                if update_guesses:
                    initial_guesses = coeffs

                df = df.append(results, ignore_index=True)
                pbar.update(1)

        self.multimodel_results = df
        return df

    def map_model_function(self, comps, initial_guesses, parallel=True, nCores=None, mapUnit=None, background=None, update_guesses=True, lsqKwargs=None):
        '''
        Alternative model function to MLLSmodel.multimodel
        Intended to be compatible with multiprocessing
        '''
        if nCores is None:
            nCores = mp.cpu_count()
        if mapUnit is None:
            mapUnit = nCores

        self.comps = comps

        points = np.prod(self.HL.data.axes_manager.shape[:-1])

        hold = np.empty(points, dtype=object)

        with tqdm(desc='Quantifying SI', total=points, unit='spectra') as pbar:
            idx = np.ndindex(self.HL.data.axes_manager.shape[:-1])

            for j in range(int(np.ceil(points/mapUnit))):

                idx_to_map = list()
                for i in range(mapUnit):
                    idx_to_map.append(next(idx))

                print(idx_to_map)

                params = (comps, initial_guesses)
                kwparams = { 'fit_background':background, 
                    'lsqKwargs':lsqKwargs }

                part_func = partial(self.model_point, *params, **kwparams)

                if parallel:
                    pool = mp.Pool(processes=nCores)
                    coeffs = pool.map(part_func, idx_to_map)
                    pool.close()
                else:
                    coeffs = list(map(part_func, idx_to_map))

                t = np.empty(len(coeffs), dtype=object)
                t[:] = coeffs
                hold[j*mapUnit:(j*mapUnit+len(idx_to_map))] = t


            pbar.update(mapUnit)

        return rtn



    def generate_element_maps(self):
        '''
        Using dataframe containing multimodel results, extract elemental
        quantities into ndarrays for ease of inspection.
        '''
        self.qmaps = dict()

        canvas = np.zeros(self.dims[:-1])

        for comp in self.comps:
            qmap = canvas.copy()
            for i in range(len(self.multimodel_results)):
                qmap[self.multimodel_results['coord'][i][::-1]] = self.multimodel_results[comp][i]
            self.qmaps[comp] = qmap

    def generate_percentage_maps(self):
        '''
        Generate composition percentages based on elemental maps created using
        generate_element_maps method.
        '''
        self.pmaps = dict()

        total = np.zeros(self.dims[:-1])

        for comp in self.comps:
            total += self.qmaps[comp]

        for comp in self.comps:
            self.pmaps[comp] = 100* self.qmaps[comp]/total



class MLLSmodel:
    '''
    Class for handling standard MLLS fitting of spectra.
    '''
    def __init__(self, core_loss, standards):
        '''
        Parameters
        ----------
        core_loss : SQuEELS Data object
            The core-loss data to be modelled.
        standards : SQuEELS Standards object
            The standards library containing the references to be used
            as fit components
        '''

        self.stds = standards
        self.HL = core_loss

        self.dims = core_loss.data.data.shape
        self.nDims = len(self.dims)
        self.sigDim = core_loss.data.axes_manager.signal_indices_in_array[0]

    def _prepare_model_data(self, idx, fit_components):
        '''
        Prepares the data required for the fitting model in a single
        numpy ndarray.  The structure of this array is based on rows
        containing:
        Row 0: Observed y-values (i.e. the spectrum)
        Row 1: x-values corresponsing to y-values
        Row 2+: The reference spectra to be fitted to the observed data.

        Params
        ------
        idx : tuple
            The indices of the spectrum within the SI to be extracted into
            the numpy array
        fit_components : list of strings
            The names of the reference spectra to be included as components
            in the fit.

        Returns
        -------
        rtn : ndarray
            An (n_comps+2) by spectrum-length numpy array
        '''
        n_comps = len(fit_components)
        x = self.HL.data.axes_manager[-1].axis

        rtn = np.empty((2+n_comps, self.dims[-1]))

        rtn[0,:] = self.HL.data.inav[idx].data
        rtn[1,:] = x
        for i in range(n_comps):
            comp = fit_components[i]
            rtn[2+i,:] = self.stds.ready[comp].inav[idx].data

        return rtn

    def _fit_model(self, data, bkgd_shape=None, lsqKwargs=None):
        '''
        Function that performs curve fitting based on use of
        scipy.optimize.curve_fit

        Params
        ------
        data : ndarray
            The data to be used in the model.  Prepared by the function
            MLLSmodel._prepare_model_data()
        bkgd_shape : string or None
            If not None, must be a string containing the name of one of
            the available background shapes:
            - power law
            - log-linear
            - double power law
        lsqKwargs : dict
            Keword arguments to be taken by scipy.optimize.curve_fit

        Returns
        -------
        opt_cofs : ndarray
            The optimised fit coefficients
        cov : ndarray
            The covariance matrix of the fit
        '''
        if lsqKwargs is None:
            lsqKwargs = {}

        y_obs = data[0,:]
        x = data[1,:]
        components = data[2:,:]

        def background(t, coeffs):
            if bkgd_shape=='power law':
                return coeffs[-2] * pow(t, coeffs[-1])
            if bkgd_shape=='log-linear':
                return np.log(coeffs[-2] * pow(t, coeffs[-1]))
            if bkgd_shape=='double power law':
                return coeffs[-4] * pow(t, coeffs[-3]) + coeffs[-2] * pow(t, coeffs[-1])

        def model(t, *coeffs):
            y = 0.0
            for i in range(components.shape[0]):
                y += coeffs[i] * components[i,:]
            if bkgd_shape:
                y += background(t, coeffs)
            return y

        opt_cofs, cov = lsq(model, x, y_obs, **lsqKwargs)

        # self.last = [opt_cofs, cov]
        return opt_cofs, cov

    def model_single_spectrum(self, index, fit_components, bkgd_shape=None, rtn_cov=False, lsqKwargs=None):
        '''
        Model the spectrum in the SI at position *index using the references
        specified in fit_components. 
        May include a background by specifying bkgd_shape.

        Params
        ------
        index : tuple
            A 2-tuple of ints specifying the inav coordinates of the spectrum
            to be modelled.
        fit_components : list of strings
            A list of the names of reference spectra to be used as components
            in the model.
        bkgd_shape : string or None
            If specified, the name of the background type to be fitted.
            See MLLSmodel._fit_model docstring for available shapes.
        rtn_cov : bool
            If true, the covariance matrix for the fit is also returned.
        lsqKwargs : dict
            Keyword arguments to be passed through to scipy.optimize.curve_fit

        Returns
        -------
        results : ndarray

        '''
        fit_data = self._prepare_model_data(index, fit_components)

        results, cov = self._fit_model(fit_data, bkgd_shape=bkgd_shape, lsqKwargs=lsqKwargs)

        if rtn_cov:
            return results, cov
        else:
            return results

    def model_full_SI(self, fit_components, bkgd_shape=None, rtn_cov=False, lsqKwargs=None):
        '''
        Model all spectra in the SI and return results.

        Params
        ------
        fit_components : list of strings
            A list of the names of reference spectra to be used as components
            in the model.
        bkgd_shape : string or None
            If specified, the name of the background type to be fitted.
            See MLLSmodel._fit_model docstring for available shapes.
        rtn_cov : bool
            If true, the covariance matrices are also stored and returned.
        lsqKwargs : dict
            Dictionary containing keword arguments to be passed to 
            scipy.optimize.curve_fit

        Returns
        -------
        rtn : ndarray
        '''
        n_points = np.prod(self.dims[:-1])

        hold = np.empty(self.dims[:-1], dtype=object)

        if rtn_cov:
            hold_cov = np.empty(self.dims[:-1], dtype=object)

        with tqdm(desc='Quantifying SI', total=n_points, unit='spectra') as pbar:
            for index in np.ndindex(self.dims[:-1]):
                fit_data = self._prepare_model_data(index, fit_components)
                results, cov = self._fit_model(fit_data, bkgd_shape=bkgd_shape, lsqKwargs=lsqKwargs)
                hold[index] = results
                if rtn_cov:
                    hold_cov[index] = cov
                pbar.update(1)

        rtn = hold
        if rtn_cov:
            cov = hold_cov
            return rtn, cov
        else:
            return rtn

    def model_full_SI_new(self, fit_components, bkgd_shape=None, 
            lsqKwargs=None, parallel=True, n_cores=None, chunk=16):
        '''
        Model all spectra in the SI and return results.

        Params
        ------
        fit_components : list of strings
            A list of the names of reference spectra to be used as components
            in the model.
        bkgd_shape : string or None
            If specified, the name of the background type to be fitted.
            See MLLSmodel._fit_model docstring for available shapes.
        rtn_cov : bool
            If true, the covariance matrices are also stored and returned.
        lsqKwargs : dict
            Dictionary containing keword arguments to be passed to 
            scipy.optimize.curve_fit
        parallel : bool
            If true, use python multiprocessing to utilise multiple CPU cores.
        n_cores : int
            The number of cores to execute the calculation on.  If None, will
            detect and use the maximum available cores.
        chunk : int
            To manage memory more efficiently, chunks of the SI are prepared and
            passed to the mapped function.  'chunk' is the max number of spectra
            to be included in an individual chunk.

        Returns
        -------
        rtn : ndarray
        '''
        if n_cores is None:
            n_cores = mp.cpu_count()

        if lsqKwargs is None:
            lsqKwargs = {'p0':np.zeros(len(fit_components))}

        n_points = np.prod(self.dims[:-1])

        hold = np.empty(n_points, dtype=object)

        with tqdm(desc='Quantifying SI', total=n_points, unit='spectra') as pbar:
            indices = np.ndindex(self.dims[:-1])

            for j in range(int(np.ceil(n_points/chunk))):
                fit_data = list()
                for i in range(chunk):
                    fit_data.append(self._prepare_model_data(next(indices), fit_components))

                kwargs = {'bkgd_shape':bkgd_shape, 'lsqKwargs':lsqKwargs}

                part_func = partial(_fit_model, **kwargs)

                if parallel:
                    pool = mp.Pool(processes=n_cores)
                    coeffs = pool.map(part_func, fit_data)
                    pool.close()
                else:
                    coeffs = list(map(part_func, fit_data))

                t = np.empty(len(coeffs), dtype=object)
                t[:] = coeffs
                hold[j*chunk:(j*chunk+len(fit_data))] = t

                pbar.update(len(fit_data))

        rtn = hold
        return rtn

    def model_full_SI_new_nochunk(self, fit_components, bkgd_shape=None, 
            lsqKwargs=None, parallel=True, n_cores=None):
        '''
        Model all spectra in the SI and return results.

        Params
        ------
        fit_components : list of strings
            A list of the names of reference spectra to be used as components
            in the model.
        bkgd_shape : string or None
            If specified, the name of the background type to be fitted.
            See MLLSmodel._fit_model docstring for available shapes.
        rtn_cov : bool
            If true, the covariance matrices are also stored and returned.
        lsqKwargs : dict
            Dictionary containing keword arguments to be passed to 
            scipy.optimize.curve_fit
        parallel : bool
            If true, use python multiprocessing to utilise multiple CPU cores.
        n_cores : int
            The number of cores to execute the calculation on.  If None, will
            detect and use the maximum available cores.

        Returns
        -------
        rtn : ndarray
        '''
        if n_cores is None:
            n_cores = mp.cpu_count()

        if lsqKwargs is None:
            lsqKwargs = {'p0':np.zeros(len(fit_components))}

        n_points = np.prod(self.dims[:-1])

        hold = np.empty(self.dims[:-1], dtype=object)

        observed = self.HL.data.data
        x = self.HL.data.axes_manager[-1].axis
        x1 = np.repeat(x[np.newaxis,:], self.dims[-2], axis=0)
        x2 = np.repeat(x1[np.newaxis,:,:], self.dims[0], axis=0)
        n_comps = len(fit_components)

        fit_data = np.empty((*self.dims[:-1], 2+n_comps, self.dims[-1]))
        fit_data[...,0,:] = observed
        fit_data[...,1,:] = x2

        for i in range(n_comps):
            comp = fit_components[i]
            fit_data[...,2+i,:] = self.stds.ready[comp].data

        chunk_axis = np.argmin(self.dims[:-1])
        non_chunk_axis = np.argmax(self.dims[:-1])

        with tqdm(desc='Quantifying SI', total=n_points, unit='spectra') as pbar:
            # indices = np.ndindex(self.dims[:-1])

            for j in range(self.dims[chunk_axis]):
                fit_chunk = list(fit_data.take(indices=j, axis=chunk_axis))

                kwargs = {'bkgd_shape':bkgd_shape, 'lsqKwargs':lsqKwargs}

                part_func = partial(_fit_model, **kwargs)

                if parallel:
                    pool = mp.Pool(processes=n_cores)
                    coeffs = pool.map(part_func, fit_chunk)
                    pool.close()
                else:
                    coeffs = list(map(part_func, fit_chunk))

                t = np.empty(len(coeffs), dtype=object)
                t[:] = coeffs
                hold.swapaxes(0, chunk_axis)[j] = t

                pbar.update(self.dims[non_chunk_axis])

        # Once fitting is complete, need to rehash data into usable
        # format.
        try:
            f = hold.ravel()
            fr, fc = zip(*f)
            fra = np.vstack(fr).reshape(hold.shape + (-1,))
            fca = np.vstack(fc).reshape(hold.shape + (-1,))

            rtn = np.rollaxis(fra, -1, 0)
            cov = np.rollaxis(fca, -1, 0)

            
        except ValueError:
            rtn = hold
            cov = 0

        return rtn, cov
