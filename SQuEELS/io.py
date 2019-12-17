from __future__ import print_function

import sys
import os

import numpy as np
import scipy as sp

from tqdm import tqdm

import hyperspy.api as hs

from .fourier_tools import fourier_ratio_convolution


class Data:
    def __init__(self, fp=None, DEELS=False, LL=None):
        '''
        Class to handle data and data operations.

        Parameters
        ----------
        fp : str
            If not None, this string is the full filepath of the data file
            to be loaded.
        DEELS : boolean
            If true, load in accompanying low-loss data also.
        '''

        if fp is None:
            from tkinter import filedialog
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            fp = filedialog.askopenfilename(filetypes=(("Gatan Files", "*.dm3"),),
                title="Choose Core-Loss Data")


        self.raw = hs.load(fp)
        self.data = self.raw
        self.info = "Raw data loaded. "
        self.sigDim = self.data.axes_manager.signal_indices_in_array[0]

        if DEELS:
            if LL is None:
                from tkinter import filedialog
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                LL = filedialog.askopenfilename(filetypes=(("Gatan Files", "*.dm3"),),
                    title="Choose accompanying Low-Loss Data")

            self.LL = hs.load(LL)
            self.LL_raw = self.LL.deepcopy()

    def normalise(self):
        '''
        Scales the maximum intensity to 1.

        '''
        self.data /= np.max(self.data.data)

        self.info += "Intensities normalised. "

    def apply_logscale(self):
        '''
        Take the natural logarithm of the data.
        Handles log issues by setting affected values to zero.
        '''
        self.data = np.log(self.data)
        self.data.data = np.nan_to_num(self.data.data, 
            copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        self.info += "Intensities log-scaled. "

    def deconvolve(self, ZLPkwargs=None, padkwargs=None):
        '''
        This method provides a hook to the deconvolution mode of the
        fourier_ratio_convolution function in SQuEELS.fourier_tools.
        Has scope to pass keword arguments to lower level functions.

        Parameters
        ----------
        ZLPkwargs : dict or None
            kwargs to be passed to extract_ZLP
        padkwargs : dict or None
            kwargs to be passed to match_spectra_sizes
        '''
        if hasattr(self, 'LL'):
            self.deconv = fourier_ratio_convolution(self.data, self.LL, 
                deconv=True, ZLPkwargs=ZLPkwargs, padkwargs=padkwargs)
            self.data = self.deconv
            self.info +=  'Fourier ratio deconvolved. '
        else:
            print('No low-loss data available for deconvolution.')

    def set_data_range(self, start, end):
        '''
            
        '''
        try:
            self.data.crop(start=start, end=end, axis=self.sigDim)
            self.info += 'Data cropped to range '+str(start)+'-'+str(end)+' eV. '
        except:
            raise Exception('Something went wrong cropping core-loss data.')

    def project_signal_on_axis(self, axis):
        '''
        Sum signal along a given axis, reducing the dimensionality of the
        dataset by 1.

        Parameters
        ----------
        axis: int
            The axis along which to sum spectra
        '''
        dims = list(self.data.axes_manager.shape)
        dims[axis] = 1
        self.data = self.data.rebin(new_shape=dims)
        self.info += 'Signal projected along axis '+str(axis)+'. '

        if hasattr(self, 'LL'):
            dims_LL = list(self.LL.axes_manager.shape)
            dims_LL[axis] = 1
            self.LL = self.LL.rebin(new_shape=dims_LL)

    def rebin_energy(self, factor):
        '''
        Rebin signal along energy axis by given factor

        Parameters
        ----------
        factor : int
            Energy rebinning factor
        '''
        dims = list(self.data.axes_manager.shape)
        dims[self.sigDim] /= factor
        self.data = self.data.rebin(new_shape=dims)
        self.info += 'Energy axis rebinned by factor of '+str(factor)+'. '
        if hasattr(self, 'LL'):
            dims_LL = list(self.LL.axes_manager.shape)
            dims_LL[self.sigDim] /= factor
            self.LL = self.LL.rebin(new_shape=dims_LL)

    def plot(self):
        '''
        Call to hyperspy SI plotting routine to save keystrokes.
        '''
        self.data.plot()

class Standards:
    def __init__(self, fp=None):
        '''
        Class to handle reference spectra from standard materials.
        Class is designed to look in one folder that contains all standards
        which the user may want to use.

        Parameters
        ----------
        fp : str
            If not None, this string is the location of the directory which
            contains the standard spectra.

        browse : Boolean
            If true, a file broswer pops up for the user to choose the
            directory where standards are stored.
            This method is ignored if a filepath fp is provided.
        
        '''

        if fp is None:
            from tkinter import filedialog
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            fp = filedialog.askdirectory()
        # Pull filenames from directory
        files = [f for f in os.listdir(fp) if os.path.isfile(os.path.join(fp, f))]

        # Load data into class
        self.data = dict()
        self.active = dict()
        for file in files:
            fn, ext = os.path.splitext(file)
            if ext == '.dm3':
                spectrum = hs.load(os.path.join(fp, file))
                self.data[fn] = spectrum
                self.active[fn] = False
            else:
                raise Exception('Unexpected file extension encountered. Expecting .dm3 files.')

        self.info = 'Reference spectra loaded. '
        # Does init need to do anything else?

    def set_active_standards(self, names):
        '''
        Sets which standards are to be used in fits.
        All standards are initially inactive by default.

        Parameters
        ----------
        names : list of strings
            List of names of standards which are to be made active.
        '''
        keys = self.active.keys()
        for name in names:
            if name in keys:
                self.active[name] = True
        self.info += 'References for '+str(names)+' set to active. '

    def set_spectrum_range(self, start, end):
        '''
        Pads/crops reference spectra to match the data range to be used in 
        the fit.

        Parameters
        ----------
        start : float
            The low energy-loss boundary of the fit in eV.
            If this lies outside the data range of a standard, the spectrum
            is padded with zeroes to fill the range.
        end : float
            The high energy-loss boundary of the fit in eV.
            If this lies outside the data range of a standard, an exception
            will be thrown.
        '''

        # Create a dictionary of standards which have been treated and ready
        # for fitting
        self.crops = dict()
        if not hasattr(self, 'ready'):
            self.ready = dict()
        # Iterate through all active standards
        for ref in self.data:
            if self.active[ref] is True:
                spec = self.data[ref].deepcopy()
                # First, crop from high energy-loss end
                try:
                    spec.crop(start=0, end=end, axis=0)
                except:
                    #raise Exception('Something went wrong cropping the high-loss end.')
                    # If cropping the high-loss end fails, print warning and
                    # pad instead.
                    print('Warning, high-loss limit extends beyond the range of reference "', ref, '".')
                    print('Spectrum will been padded, which may impact quantification.')
                    print('')
                    # nPad = int(np.round(end - ))
                # Next, crop/pad low-loss end
                offset = spec.axes_manager[0].offset
                if start < offset:
                    # Pad spectrum out
                    nPad = int(np.round((offset - start)/spec.axes_manager[0].scale))
                    new = np.zeros((nPad+spec.axes_manager[0].size))
                    new[nPad:] = spec.data
                    spec.data = new
                    spec.axes_manager[0].offset = start
                    spec.axes_manager[0].size = len(new)
                else:
                    # Crop spectrum down
                    # spec.crop(start=start, end=end, axis=0)
                    spec = spec.isig[start:]
                # Add spectrum to ready list
                self.crops[ref] = spec
                self.ready[ref] = spec
                self.lims = (start, end)
                self.nDat = spec.axes_manager[0].size

        self.info += 'Reference energy range set to '+str(start)+'-'+str(end)+' eV. '

    def map_refs_to_nav(self, data):
        '''

        '''
        self.mapped = dict()

        dims = list(data.data.shape)
        dims[-1] = self.nDat # Determined during set_spectrum_range

        for ref in self.ready:
            mapped = data.deepcopy()
            mapped.data = np.zeros(dims)
            mapped.axes_manager[-1].offset = self.lims[0]
            mapped.axes_manager[-1].size = self.nDat
            mapped.data[...,:] = self.ready[ref].data
            self.mapped[ref] = mapped

        self.ready = self.mapped

        self.info += 'Spectra mapped onto data navigation dimensions. '

    def normalise(self, logscale=False):
        '''
        Scales the integrated intensity of range-set references to 1.

        Should only be used in conjunction with set_spectrum_range.

        Parameters
        ----------
        logscale : Boolean
            Sets whether normalisation of references is preceded by 
            taking the natural logarithm.
        '''
        self.normed = dict()
        self.norm_coeffs = dict()

        if hasattr(self, 'ready'):
            for ref in self.ready:
                spec = self.ready[ref].deepcopy()
                if logscale:
                    spec = np.log(spec)
                    spec.data = np.nan_to_num(spec.data, 
                        copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                    spec.data[spec.data<0.0] = 0.0
                scale_factor = np.max(spec.data)
                spec /= scale_factor
                # Write the scaled reference, along with factor, to class
                self.normed[ref] = spec
                self.norm_coeffs[ref] = scale_factor
                self.ready[ref] = spec
            self.info += 'Reference spectra normalised with logscale set to '+str(logscale)+'. '
        else:
            raise Exception("Cropped reference spectra not found.")

    def set_all_inactive(self):
        '''
        Sets all standards in library to active=False
        '''
        for item in self.active:
            self.active[item] = False

    def convolve_mapped(self, LL, ZLPkwargs=None, padkwargs=None):
        '''
        Convolve the prepared reference spectra with a low loss spectrum.
        If using a spectrum image, this generates a map of the same spatial
        dimensions using each reference, so that the deconvolution is unique
        to each spectrum in the low-loss.

        Parameters
        ----------
        LL : Hyperspy signal object
            The low-loss spectrum the core-loss edges are to be convolved with   
        ZLPkwargs : dict or None
            keyword arguments for the ZLP extraction.
        padkwargs : dict or None
            keyword arguments for spectral padding.
        '''

        self.conv = dict()
        # Get dimensions of signal

        with tqdm(total=len(self.ready), unit='standards') as pbar:
            for ref in self.ready:
                self.conv[ref] = fourier_ratio_convolution(self.mapped[ref], LL, False, 
                    ZLPkwargs=ZLPkwargs, padkwargs=padkwargs)
                pbar.update(1)

        self.ready = self.conv
        self.info += 'Mapped references convolved with data low-loss. '

    def project_map_on_axis(self, axis):
        '''
        Sum signal along a given axis, reducing the dimensionality of the
        dataset by 1.

        Parameters
        ----------
        axis: int
            The axis along which to sum spectra
        '''
        self.projected = dict()

        for comp in self.ready:

            dims = list(self.ready[comp].axes_manager.shape)
            dims[axis] = 1
            self.projected[comp] = self.ready[comp].rebin(new_shape=dims)

        self.ready = self.projected

        self.info += 'Mapped references projected onto axis '+str(axis)+'. '

    def rebin_energy(self, factor):
        '''

        '''
        self.rebinned = dict()

        for comp in self.ready:
            dims = list(self.ready[comp].axes_manager.shape)
            dims[-1] /= factor
            self.rebinned[comp] = self.ready[comp].rebin(new_shape=dims)

        self.info += 'Reference spectra rebinned in energy by factor '+str(factor)+'. '
        self.ready = self.rebinned
