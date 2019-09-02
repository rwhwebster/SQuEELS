from __future__ import print_function

import sys
import os


import numpy as np
import scipy as sp

import hyperspy.api as hs

def dummy_io(a):
    return a

class standards:
    def __init__(self, fp=None, browse=True):
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
            This method is skipped if a filepath fp is provided.
        
        '''

        if fp is None and browse is True:
            from tkinter import filedialog
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            fp = filedialog.askdirectory()
        # Pull filenames from directory
        files = [f for f in os.listdir(fp) if os.path.isfile(os.path.join(fp, f))]

        # Load data into class
        self.data = dict()
        for file in files:
            fn, ext = os.path.splitext(file)
            if ext == '.dm3':
                spectrum = hs.load(os.path.join(fp, file))
                self.data[fn] = spectrum
            else:
                raise Exception('Unexpected file extension encountered. Expecting .dm3 files.')

        # Does init need to do anything else?



