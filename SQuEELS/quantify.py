from __future__ import print_function

import numpy as np
import scipy as sp

from tqdm import tqdm

import hyperspy.api as hs

import matplotlib.pyplot as plt
plt.ion()

class LRmodel:
    '''
    Class for handling quantification of data using simple linear regression.
    '''
    def __init__(self, core_loss, stds, comps, data_range, low_loss=None):
        '''
        Create a Linear Regression Model.

        Parameters
        ----------
        core_loss : Hyperspy spectrum or spectrum image

        stds : SQuEELS Standards object

        comps : list of strings

        data_range : tuple of floats

        low_loss : Hyperspy spectrum or spectrum image

        '''
        self.stds = stds
        self.comps = comps

        self.stds.set_all_inactive()
        self.stds.set_active_standards(self.comps)

        if low_loss:
            self.LL = low_loss
        # Check how many dimensions there are and crop data to fit range
        self.dims = core_loss.data.shape
        self.nDims = len(self.dims)
        sigDim = core_loss.axes_manager.signal_indices_in_array[0]
        try:
            self.HL = core_loss.deepcopy()
            self.HL.crop(start=data_range[0], end=data_range[1], axis=sigDim)
        except:
            raise Exception("Cropping of observed signal failed.")
        # If successful, crop/pad standards to same range
        self.stds.set_spectrum_range(*data_range)

    def _init_step(self, nav):
        '''
        Perform any preparatory steps before performing fit.
        '''
        if self.LL:
            currentL = self.LL.inav[nav[0], nav[1]]
            self.stds.convolve_ready(currentL, kwargs={'stray':True})
            model = self.stds.conv
        else:
            model = self.stds.ready
        y_obs = self.HL.inav[nav[0], nav[1]].data.T
        x_mat = np.array([model[comp].data for comp in self.comps]).T

        return y_obs, x_mat

    def do_multifit(self):
        '''
        Run regression on all spectra in dataset
        '''
        d1 = self.dims[0]
        d2 = self.dims[1]
        self.results = np.zeros((d1, d2, len(self.comps)))
        with tqdm(total=(d1*d2), unit='spectra') as pbar:
            for i in range(d1):
                for j in range(d2):
                    Y, X = self._init_step((j, i))
                    self.results[i,j,:] = _normalEqn(X, Y)
                    pbar.update(1)
        return 1


    def plot_multifit_results(self, cmap='viridis'):
        '''
        Plot fit coefficient maps.
        '''
        nComps = len(self.comps)
        nRows = np.int_(np.floor(np.sqrt(nComps)))
        nCols = np.int_(np.ceil(nComps/nRows))

        fig, subplots = plt.subplots(nRows, nCols)
        for i, ax in enumerate(fig.axes):
            if i in range(nComps):
                ax.imshow(self.results[:,:,i], cmap=cmap) 
                # plt.matplotlib.colors.SymLogNorm(linthresh=0.03)
                ax.set_title(self.comps[i])
                ax.axis('off')
        fig.suptitle("Regression Maps")
        fig.show()


# END OF CLASS


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
