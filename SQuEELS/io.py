from __future__ import print_function

import sys
import os


import numpy as np
import scipy as sp

import hyperspy.api as hs

from .fourier_tools import reverse_fourier_ratio_convoln


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
        if not self.ready:
            self.ready = dict()
        # Iterate through all active standards
        for ref in self.data:
            if self.active[ref] is True:
                spec = self.data[ref].deepcopy()
                # First, crop from high energy-loss end
                try:
                    spec.crop(start=0, end=end, axis=0)
                except:
                    raise Exception('Something went wrong cropping the high-loss end.')
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

        if self.ready not None:
            for ref in self.ready[ref]:
                spec = ref.deepcopy()
                if logscale:
                    spec = np.log(spec)
                scale_factor = np.sum(spec.data)
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

    def convolve_ready(self, LL, kwargs=None):
        '''
        Convolve the prepared reference spectra with a low loss spectrum.

        Parameters
        ----------
        LL : Hyperspy spectrum object
            The low-loss spectrum the core-loss edges are to be convolved with   
        kwargs : dict or None
            keyword arguments for the convolution function.
        '''
        if not kwargs:
            kwargs = {}

        self.conv = dict()

        for ref in self.ready:
            self.conv[ref] = reverse_fourier_ratio_convoln(self.ready[ref], LL, **kwargs)
