import os
import sys

import numpy as np
import scipy as sp

import matplotlib.pylab as plt

import hyperspy.api as hs

import pymc3 as pm

import SQuEELS as sq

plt.ion()

refs = sq.io.Standards(fp='/mnt/d/RWebster/Lab_Data/Experimental_EELS_Cross_Sections/SQuEELS', browse=False)

# Create Dummy Spectrum

refs.set_active_standards(['Ti','Ni','Sn'])

refs.set_spectrum_range(350.0, 1050.0)

test = refs.ready['Ti'].deepcopy()*0

for comp in refs.ready:
    test += refs.ready[comp]*10

new_model = sq.quantify.create_bayes_model(refs, ['Ti','Ni','Sn'], test, (400.0, 1000.0), guess=11.0, plot=True)

map_estimate = pm.find_MAP(model=new_model)

with new_model:
    trace = pm.sample(1000, tune=1000, cores=4)

with new_model:
    step = pm.Slice()
    trace = pm.sample(5000, step=step)