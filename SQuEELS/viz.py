from __future__ import print_function

import os
import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
plt.ion()

class FitInspector:
    '''
    Class for handling visualisation actions on model results.

    '''
    def __init__(self, data, stds, coeffs, data_range, low_loss=None):
        '''
        
        '''
        