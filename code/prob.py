'''Likelihood function.'''
import numpy as np
import collections
from . import utils
from gsm import gsm  # gaussian streaming model


class prob:
    '''Likelihood function class.'''

    def __init__(self, fn):
        '''parse the ini file'''

        self.pars = collections.OrderedDict()
        self.pars['keys'] = ['nu', 'f', 'sFOG']

        self.inis = collections.OrderedDict()
        self.inis['keys'] = ['xi0file', 'xi2file', 'cov_file',
                             'clpt_xi', 'clpt_v', 'clpt_s',
                             'slim']

        # load & parse the ini file
        lines = [line.strip() for line in open(fn).readlines()]
        for line in lines:
            if len(line) == 0:  # empty line
                continue
            if line[0] == '#':  # comment
                continue
            ls = line.split()
            if ls[0] in self.pars['keys']:
                self.pars[ls[0]] = np.array(ls[1:]).astype(np.float)
            elif ls[0] in self.inis['keys']:
                if ls[0] == 'slim':
                    self.inis[ls[0]] = np.array(ls[1:]).astype(np.float)
                else:
                    self.inis[ls[0]] = ls[1]

        # make data vector & covariance matrix
        xi0 = np.loadtxt(self.inis['xi0file'])
        xi2 = np.loadtxt(self.inis['xi2file'])
        idx = (xi0[:, 0] >= self.inis['slim'][0]) \
            & (xi0[:, 0] <= self.inis['slim'][1])
        xi02 = np.row_stack((xi0[idx], xi2[idx]))
        self.ss = xi0[:, 0][idx]  # s sample points
        self.data = xi02[:, 1]

        if self.inis['cov_file'] == 'jackknife':
            self.cov = utils.jackknife_cov(xi02[:, 4:])
        else:
            self.cov = np.loadtxt(self.inis['cov_file'])

        self.icov = np.linalg.inv(self.cov)

        # model related
        self.gsrsd = gsm.gsm()
        self.gsrsd.read_clpt(fn_xi=self.inis['clpt_xi'], fn_v=self.inis['clpt_v'],
                             fn_s=self.inis['clpt_s'])
        self.gsrsd.set_s_mu_sample(self.ss)
        self.gsrsd.set_y_sample()

        self.npars = len(self.pars['keys'])  # number of parameters
        self.ndata = len(self.data)  # number of data points

        self.dof = self.ndata - self.npars + 1  # degree of freedom

    def ini_pars(self, nwalkers):
        '''initialize the walkers'''
        # uniform distribution
        pos0 = np.random.random((nwalkers, self.npars))
        for i, p in enumerate(self.pars['keys']):
            width = self.pars[p][2] - self.pars[p][1]
            pos0[:, i] = self.pars[p][1] + width * pos0[:, i]

        return pos0

    def c_model(self, theta):
        '''compute the model vector'''
        self.gsrsd.set_pars(nu=theta[0], f_v=theta[1], sFOG=theta[2])
        xi0, xi2 = self.gsrsd.c_xi()
        return np.concatenate((xi0, xi2))

    def ln_prior(self, theta):
        '''ln prior'''
        # uniform in the limited range
        for i, par in enumerate(self.pars['keys']):
            if theta[i] < self.pars[par][1] or theta[i] > self.pars[par][2]:
                return -np.inf
        return 0.

    def ln_likelihood(self, theta):
        '''ln likelihood'''
        diff = self.data - self.c_model(theta)
        chi2 = np.einsum('i,ij,j', diff, self.icov, diff)
        return -0.5 * chi2

    def ln_prob(self, theta):
        '''ln probability'''
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.ln_likelihood(theta)
