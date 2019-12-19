from __future__ import print_function

import numpy as np
import scipy as sp
from scipy.optimize import leastsq

import pandas as pd

from tqdm import tqdm

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()


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

    def model_point(self, coords, comps, init_guess, fit_background=None):
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
        def background(t, coeffs):
            if fit_background=='power law':
                return coeffs[-2] * pow(t, coeffs[-1])
            if fit_background=='log-linear':
                return np.log(coeffs[-2] * pow(t, coeffs[-1]))
            if fit_background=='double power law':
                return coeffs[-4] * pow(t, coeffs[-3]) + coeffs[-2] * pow(t, coeffs[-1])

        def model(t, coeffs):
            y = 0.0
            for i, comp in enumerate(comps):
                y += coeffs[i] * self.stds.ready[comp].inav[coords].data
            if fit_background:
                y += background(t, coeffs)
            return y

        def residuals(coeffs, yObs, t):
                return yObs - model(t, coeffs)

        y_Obs = self.HL.data.inav[coords].data
        disp = self.HL.data.axes_manager[self.sigDim].scale
        offset = self.HL.data.axes_manager[self.sigDim].offset
        nDat = self.HL.data.axes_manager[self.sigDim].size
        t = np.linspace(offset, offset+(disp*nDat), nDat)

        coefficients, flag = leastsq(residuals, init_guess, args=(y_Obs, t))

        self.last = coefficients
        return coefficients

    def multimodel(self, comps, initial_guesses, background=None, update_guesses=True):
        '''

        '''
        self.comps = comps
        df = pd.DataFrame(columns=['coord',*comps])

        points = np.prod(self.HL.data.axes_manager.shape[:-1])

        with tqdm(desc='Quantifying SI', total=points, unit='spectra') as pbar:
            for idx in np.ndindex(self.HL.data.axes_manager.shape[:-1]):
                results = {'coord': idx, }

                coeffs = self.model_point(idx, comps, initial_guesses, fit_background=background)
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