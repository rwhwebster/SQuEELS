from __future__ import print_function

import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ion()

_mpl_non_adjust = False
_mplv = mpl.__version__
from distutils.version import LooseVersion
if LooseVersion(_mplv) >= LooseVersion('2.2.0'):
    _mpl_non_adjust = True

def _power_law(x, coeffs):
    '''
    Provides a power law background shape for the FitInspector class.

    Parameters
    ----------
    x : ndarray
        The x values to be used to calculate corresponding y values
    coeffs : ndarray
        The array of coefficients for all the components of a SQuEELS MLLS
        fit.  Assumes the last two components are for the power law background.

    Returns
    -------
    background : ndarray
        Array with the spatial dimensions of the coeffs map and the z dimension
        length of the x values.  Contains the calculated power law shape.
    '''
    background = np.empty(coeffs.shape[1:]+(len(x),))
    background[...,:] = x[np.newaxis, np.newaxis, :]
    background = coeffs[-2,...,np.newaxis] * pow(background, coeffs[-1,...,np.newaxis])
    return background


class FitInspector:
    def __init__(self, data, fit_components, refs, coeffs, 
                 nav_im=None, bkgd_model=None):
        '''
        Navigate results of model fitting of spectrum image.

        Very heavily inspired by the DataBrowser class in fpd 
        (https://gitlab.com/fpdpy/fpd).  Credit to GWP.

        Note that any operations such as data-ranging and convolution
        are assumed to have already been performed on the inputs.

        Parameters
        ----------
        data : hyperspy Signal object
            The spectrum image to inspect
        fit_components : list of strings

        refs : dict
            The active fit components contained in a SQuEELS Standards object
        coeffs : ndarray
            The fit coefficients which scale the components to match the data
        nav_im : 2-dimensional ndarray
            
        bkgd_model : callable?

        '''

        self.data = data.data
        self.e_ax = data.axes_manager[-1].axis
        self.ndat = len(self.e_ax)

        self.scanY, self.scanX = self.data.shape[:-1]

        # Prepare fit data
        self.coeffs = coeffs
        self.comp_names = fit_components
        self.n_comps = len(self.comp_names)
        self.comps = np.empty(self.n_comps, dtype=object)
        for i, comp in enumerate(self.comp_names):
            self.comps[i] = refs[comp].data
            self.comps[i] = (self.comps[i].T * self.coeffs[i].T).T

        self.bkgd = np.zeros(self.data.shape)
        if bkgd_model is not None:
            self.bkgd = bkgd_model(self.e_ax, self.coeffs)

        # combine fit data into single array
        self.fit_data = np.empty((self.scanY, self.scanX, self.n_comps+2, self.ndat))
        self.fit_data[...,0,:] = self.data
        self.fit_data[...,1,:] = self.bkgd
        for i in range(self.n_comps):
            self.fit_data[...,2+i,:] = self.comps[i]

        # Prepare navigation image
        self.nav_im = nav_im
        if self.nav_im is None:
            self.nav_im = self.data.sum(-1)

        self.scanYind = 0
        self.scanXind = 0
        self.scanYind_old = self.scanYind
        self.scanXind_old = self.scanXind

        self.plot_data = self.fit_data[self.scanYind, self.scanXind, ...]

        self.rwh = max(self.scanY, self.scanX)//64
        if self.rwh == 0:
            self.rwh = 2
        self.rect = None
        self.press = None
        self.background = None
        self.plot_nav_im()

        self.plot_spectrum()
        self.connect()

    def plot_nav_im(self):
        self.f_nav, ax = plt.subplots()
        self.f_nav.canvas.set_window_title('nav')

        d = {'cmap':'gray'}

        im = ax.imshow(self.nav_im, interpolation='nearest', **d)
        # plt.colorbar(mappable=im)

        rect_c = 'r'
        self.rect = mpl.patches.Rectangle((self.scanYind - self.rwh/2,
                                           self.scanXind - self.rwh/2),
                                           self.rwh, self.rwh, ec=rect_c,
                                           fc='none', lw=2, picker=4)

        ax.add_patch(self.rect)
        plt.tight_layout()
        plt.draw()

    def plot_spectrum(self):
        self.f_spec, ax = plt.subplots()
        self.f_spec.canvas.set_window_title('spectrum')

        self.im = ax.plot(self.e_ax, self.plot_data[0])
        for i in range(self.plot_data.shape[0]-1):
            self.im += ax.plot(self.e_ax, np.sum(self.plot_data[1:2+i], axis=0))

        plt.sca(ax)
        ax.format_coord = self.format_coord
        self.update_spec_plot()

        self.leg_list = ['Observed Data', 'Background'] + self.comp_names

        ax.legend(self.im, self.leg_list, loc='upper right')

        plt.tight_layout()
        plt.draw()

    def connect(self):
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.cid_f_nav = self.f_nav.canvas.mpl_connect('close_event', self.handle_close)
        self.cid_f_spec = self.f_spec.canvas.mpl_connect('close_event', self.handle_close)

    def handle_close(self, e):
        self.disconnect()
        if e.canvas.get_window_title()=='nav':
            plt.close(self.f_spec)
        else:
            plt.close(self.f_nav)

    def on_press(self, event):
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if contains:
            x0, y0 = self.rect.xy

            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            self.rect.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

            axes.draw_artist(self.rect)
            canvas.blit(axes.bbox)
        else:
            x0, y0 = (None,)*2

        self.press = x0, y0, event.xdata, event.ydata
        self.yind_temp = self.scanYind
        self.xind_temp = self.scanXind

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        if self.background is None: return

        x0, y0, xpress, ypress = self.press
        dx = int(event.xdata - xpress)
        dy = int(event.ydata - ypress)
        if abs(dy)>0 or abs(dx)>0:
            self.rect.set_x(x0+dx)
            self.rect.set_y(y0+dy)
            self.scanYind = self.yind_temp+dy
            self.scanXind = self.xind_temp+dx

            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            canvas.restore_region(self.background)
            axes.draw_artist(self.rect)
            canvas.blit(axes.bbox)
            self.update_spec_plot()

    def on_release(self, event):
        if event.inaxes != self.rect.axes: return

        x, y = self.press[2:]
        if np.round(event.xdata-x)==0 and np.round(event.ydata-y)==0:
            # mouse didn't move
            x, y = np.round(x), np.round(y)
            self.rect.set_x(x-self.rwh/2)
            self.rect.set_y(y-self.rwh/2)
            self.scanYind = int(y)
            self.scanXind = int(x)
                    
            #self.rect.figure.canvas.draw()
            self.update_spec_plot()
        elif self.background is not None:
            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            canvas.restore_region(self.background)  # restore the background region
            axes.draw_artist(self.rect)             # redraw just the current rectangle
            canvas.blit(axes.bbox)                  # blit just the redrawn area
            
        'on release we reset the press data'
        self.press = None
        self.yind_temp = None
        self.yind_temp = None
        
        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def format_coord(self, x, y):
        return 'x=%d, y=%d'%(x, y)

    def update_spec_plot(self):
        self.plot_data = self.fit_data[self.scanYind, self.scanXind, ...]

        for i in range(self.plot_data.shape[0]):
            if i==0:
                self.im[i].set_ydata(self.plot_data[i])
                self.im[0].axes.set_ylim((self.plot_data[:2].min(), self.plot_data[:2].max()))
            else:
                self.im[i].set_ydata(np.sum(self.plot_data[1:1+i], axis=0))
            self.im[i].axes.figure.canvas.draw()


    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
        
        self.f_nav.canvas.mpl_disconnect(self.cid_f_nav)
        self.f_spec.canvas.mpl_disconnect(self.cid_f_spec)
