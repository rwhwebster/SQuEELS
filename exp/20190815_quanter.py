import os
import sys

import numpy as np
import scipy as sp

import matplotlib.pylab as plt

import hyperspy.api as hs

import SQuEELS as sq

plt.ion()

# Load EELS Standards

refdir = '/mnt/d/RWebster/Lab_Data/Experimental_EELS_Cross_Sections/20180405_New_Fourier_Ratio_Standards/'

os.chdir(refdir)

Ti = hs.load('Ti DXC (FR) (SI01).dm3')
Ni = hs.load('Ni DXC (FR) (SI07).dm3')
Sn = hs.load('Sn DXC (FR).dm3')
O = hs.load('O DXC (FR).dm3')
Cu = hs.load('Cu DXC (FR) (SI01).dm3')






# Load Experimental Data to be Fitted

datadir = '/mnt/d/RWebster/Lab_Data/ARM/2019/20190402_SB042C_SPED_EELS/Processing/EELS_Processing/SI06/'

os.chdir(datadir)

LL = hs.load('SI_06_EELS Spectrum Image (low-loss) (CorrE)(CorrI) (aligned).dm3')
HL = hs.load('SI_06_EELS Spectrum Image (high-loss) (CorrE)(CorrI) (aligned).dm3')
HLC = hs.load('06_high-loss (signal) (FRD) (rescaled).dm3')

# Load Carbon data for checks
carbdir = '/mnt/d/RWebster/Lab_Data/ARM/2019/20190710_Diamond_Reference/processing/0p25/'

os.chdir(carbdir)

DLL = hs.load('EELS_0031 (CorrE)(CorrI) (Aligned).dm3')
DHL = hs.load('EELS_0032 (CorrE)(CorrI) (Aligned) (backsub).dm3')


# Define some functions to prepare data for quantification

def extend_spectrum(s, d1, d2):
    '''
    Attempt at mimicing 'cosine bell extrapolation' mentioned in DM
    '''

    delta = d2 - d1

    hann = np.zeros((delta))

    for i in range(delta):
        hann[i] = np.cos(np.pi*i/delta + np.pi)
        # pad[i] = np.cos((i/delta)*np.pi/2)

    hann = 0.5 - 0.5 * hann

    out = s.deepcopy()
    out.axes_manager[0].size = d2

    a0 = np.mean(s.data[-20:])

    pad = a0 * hann

    out.data = np.zeros((d2))
    out.data[:d1] = s.data
    out.data[d1:] = pad

    return out


def pad_spectra(s1, s2):
    '''
    Pads spectra to common size and extrapolates through empty space
    '''
    l1 = len(s1.data)
    l2 = len(s2.data)
    # Determine smallest 2^N that satisfies the sizes of both inputs
    n1 = int(np.ceil(np.log2(l1)))
    n2 = int(np.ceil(np.log2(l2)))
    N = max(n1,n2)+2

    k = 2**N

    o1 = extend_spectrum(s1, l1, k)
    o2 = extend_spectrum(s2, l2, k)

    return o1, o2


def clip_LL(s, reg):
    '''
    Pre-treatment for the Low-loss spectrum to ensure that the zero loss 
    '''
    xo = s.axes_manager[0].offset
    xs = s.axes_manager[0].scale
    xp = abs(xo/xs) # Channel position of zero energy
    # 
    offset = np.mean(s.data[0:int(reg*xp)])

    clip = s - offset

    clip.data[0:int(reg*xp)] = 0.0

    return clip


def reverse_fourier_ratio(HL, LL):
    '''
    Function which works through fourier ratio deconvolution
    backwards, convolving a single scattering disteribution with
    the provided low loss spectrum.
    '''

    LL = clip_LL(LL, 0.8)

    shift = HL.axes_manager[0].offset

    low, high = pad_spectra(LL, HL)

    ZLP = extract_ZLP(low)

    # Calculate Fourier Transforms
    LLF = low.fft()
    HLF = high.fft()
    ZLF = ZLP.fft()

    # Perform Convolution
    conv = (HLF*LLF)/ZLF
    # Inverse fourier transform to get convolved spectrum
    reconv = conv.ifft()

    reconv.axes_manager[0].offset = shift # Correct energy scale

    return reconv





# Create Model
m = HLC.create_model()

# Create Components
mTi = hs.model.components1D.ScalableFixedPattern(Ti)
mTi.name = 'Ti'
mNi = hs.model.components1D.ScalableFixedPattern(Ni)
mNi.name = 'Ni'
mSn = hs.model.components1D.ScalableFixedPattern(Sn)
mSn.name = 'Sn'
mO = hs.model.components1D.ScalableFixedPattern(O)
mO.name = 'O'
mCu = hs.model.components1D.ScalableFixedPattern(Cu)
mCu.name = 'Cu'

comps = [mTi, mNi, mSn, mCu, mO]

# Remove the automatically added Power Law
m.remove(0)

# Add Components To Model
for component in comps:
    m.append(component)


# Set Constraints
m.set_parameters_not_free()
m.set_parameters_free(parameter_name_list=['yscale'])
m.set_signal_range(400.0, 1000.0)

# Set lower bound at zero
for component in m:
    if component.name is not 'PowerLaw':
        component.yscale.bmin = 0.0
        component.yscale.bounded = True




m.fit(fitter='mpfit', bounded=True)


m.multifit(show_progressbar=True, fitter='mpfit', bounded=True)





LLtest = LL.inav[9,50]
HLtest = HL.inav[9,50]

TiC = reverse_fourier_ratio(Ti, LLtest)
NiC = reverse_fourier_ratio(Ni, LLtest)
SnC = reverse_fourier_ratio(Sn, LLtest)
OC = reverse_fourier_ratio(O, LLtest)
CuC = reverse_fourier_ratio(Cu, LLtest)

m = HLtest.create_model()

mTi = hs.model.components1D.ScalableFixedPattern(TiC)
mTi.name = 'Ti'
mNi = hs.model.components1D.ScalableFixedPattern(NiC)
mNi.name = 'Ni'
mSn = hs.model.components1D.ScalableFixedPattern(SnC)
mSn.name = 'Sn'
mO = hs.model.components1D.ScalableFixedPattern(OC)
mO.name = 'O'
mCu = hs.model.components1D.ScalableFixedPattern(CuC)
mCu.name = 'Cu'

comps = [mTi, mNi, mSn, mCu, mO]


# Add Components To Model
for component in comps:
    m.append(component)


# Set Constraints
m.set_parameters_not_free()
m.set_parameters_free(parameter_name_list=['yscale','A','r'])
m.set_signal_range(400.0, 1000.0)

# Set lower bound at zero
for component in m:
    if component.name is not 'PowerLaw':
        component.yscale.bmin = 0.0
        component.yscale.bounded = True



m.fit(fitter='mpfit', bounded=True)

# second stage refinement
m.set_parameters_free(parameter_name_list=['shift'])
for component in m:
    if component.name is not 'PowerLaw':
        component.shift.bmin = -2.5
        component.shift.bmax = 5.0
        component.shift.bounded = True

m.fit(fitter='mpfit', bounded=True)