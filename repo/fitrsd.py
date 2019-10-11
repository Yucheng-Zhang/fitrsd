'''
MCMC fitting of RSD.
'''
import emcee
import numpy as np
from . import cosmo_pars
import collections


class fitrsd:
    '''MCMC fitting of RSD w/ monopole and quadrupole.'''

    def __init__(self):

        # fiducial cosmology
        self.cosmo = None
        # monopole and quadrupole data
        self.data = None
        # inverse covariance matrix of data
        self.icov = None
        # redshift
        self.z = None

        # number of parameters to fit
        self.npars = None
        self.pars = None

        self.sampler = None

    def set_cosmology(self, tag='planck18'):
        '''Set the fiducial cosmology.'''
        self.cosmo = cosmo_pars.get_cosmology(tag)

    def set_redshift(self, z):
        '''Set the effective redshift.'''
        self.z = z

    def ini_data(self, fn0, fn2, s_min, s_max):
        '''Initialize the data to fit.'''
        print('>> xi files: {0:s}, {1:s}'.format(fn0, fn2))
        xi0, xi2 = np.loadtxt(fn0), np.loadtxt(fn2)
        # cut in separation distance
        idx = np.where((xi0[:, 0] >= s_min) & (xi0[:, 0] <= s_max))[0]
        xi02 = np.row_stack((xi0[idx], xi2[idx]))
        # data & covariance w/ jackknife
        self.data = xi02[:, :2]
        njk = xi02.shape[1] - 4
        cov = (njk - 1.) * np.cov(xi02[:, 4:], ddof=0)
        self.icov = np.linalg.inv(cov)

        print('> dof : {0:d}'.format(self.data.shape[0]))

    def ini_pars(self, pars=['nu', 'f', 'sFOG'],):
        '''Initialize the parameters.'''
        self.pars = pars
        self.npars = len(pars)

    def c_model(self):
        '''Compute the model values.'''

    def lnlike(self):
        '''ln likelihood function.'''
        # get the model

        # compute chi2

    def run_mcmc(self, nwalkers=100):
        '''Run MCMC.'''
        def lnprob(theta):
            pass

        sampler = emcee.EnsembleSampler(nwalkers, self.npars, lnprob)
