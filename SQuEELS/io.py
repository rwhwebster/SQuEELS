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
        else:
            raise Exception("Cropped reference spectra not found.")

    def set_all_inactive(self):
        '''
        Sets all standards in library to active=False
        '''
        for item in self.active:
            self.active[item] = False

    def convolve_ready(self, LL, ZLPkwargs=None, padkwargs=None):
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
        self.mapped = dict()
        # Get dimensions of LL signal
        dims = list(LL.data.shape)
        dims[-1] = self.nDat # Determined during set_spectrum_range

        for ref in self.ready:
            mapped = LL.deepcopy()
            mapped.data = np.zeros(dims)
            mapped.axes_manager[-1].offset = self.lims[0]
            mapped.axes_manager[-1].size = self.nDat
            mapped.data[...,:] = self.ready[ref].data
            self.mapped[ref] = mapped

        with tqdm(total=len(self.ready), unit='standards') as pbar:
            for ref in self.ready:
                self.conv[ref] = fourier_ratio_convolution(self.mapped[ref], LL, False, 
                    ZLPkwargs=ZLPkwargs, padkwargs=padkwargs)
                pbar.update(1)


