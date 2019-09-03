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

mult = 10

for comp in refs.ready:
    test += refs.ready[comp]*mult

test += np.random.randn(len(test.data))*10

new_model = sq.quantify.create_bayes_model(refs, ['Ti','Ni','Sn'], test, (400.0, 1000.0), guesses=(10.5,9.5,10.1), plot=True)

map_estimate = pm.find_MAP(model=new_model)

with new_model:
    trace = pm.sample(500)

with new_model:
    trace = pm.sample(1000, tune=1000, cores=4)

with new_model:
    step = pm.Slice()
    trace = pm.sample(5000, step=step)

sq.quantify.fit_bayes_model(model=new_model, nDraws=1000, params={'tune':1000, 'cores':4, 'chains':8}, plot=True)


# test fitting convolved spectrum

HL = hs.load('/mnt/d/RWebster/Lab_Data/ARM/JH Bulk/20180521/Processing/high-loss (signal).dm3')

LL = hs.load('/mnt/d/RWebster/Lab_Data/ARM/JH Bulk/20180521/Processing/1_EELS Spectrum Image (low-loss) (CorrE)(CorrI) (aligned).dm3')

HLtest = HL.inav[17,8]
LLtest = LL.inav[17,8]