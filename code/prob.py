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
        self.pars['keys'] = ['nu', 'f', 'sFOG']  # all the possible parameters
        self.pars['pars'] = []  # fitting parameters

        self.inis = collections.OrderedDict()
        self.inis['keys'] = ['xi0file', 'xi2file', 'icov_file',
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
            if ls[0] in self.pars['keys']:  # paramters
                self.pars[ls[0]] = np.array(ls[1:]).astype(np.float)
                if self.pars[ls[0]][0] > self.pars[ls[0]][1] and \
                        self.pars[ls[0]][0] < self.pars[ls[0]][2]:  # fitting parameter
                    self.pars['pars'].append(ls[0])

            elif ls[0] in self.inis['keys']:
                if ls[0] == 'slim':
                    self.inis[ls[0]] = np.array(ls[1:]).astype(np.float)
                else:
                    self.inis[ls[0]] = ls[1]

        # output pars
        str_ = 'fitting parameters: '
        for k_ in self.pars['pars']:
            str_ += '  {0:s}'.format(k_)
        str_ += '\n fixed parameters: '
        for k_ in self.pars['keys']:
            if k_ not in self.pars['pars']:
                str_ += '  {0:s}'.format(k_)
        print(str_)

        self.npars = len(self.pars['pars'])  # number of parameters

    def ini_dm(self):
        '''prepare data and model'''
        # make data vector & inverse covariance matrix
        xi0 = np.loadtxt(self.inis['xi0file'])
        xi2 = np.loadtxt(self.inis['xi2file'])
        idx = (xi0[:, 0] >= self.inis['slim'][0]) \
            & (xi0[:, 0] <= self.inis['slim'][1])
        xi02 = np.row_stack((xi0[idx], xi2[idx]))
        self.ss = xi0[:, 0][idx]  # s sample points
        self.data = xi02[:, 1]

        if self.inis['icov_file'] == 'jackknife':
            self.icov = utils.jackknife_icov(xi02[:, 4:])
        else:
            self.icov = np.loadtxt(self.inis['icov_file'])

        # model related
        self.gsrsd = gsm.gsm()
        self.gsrsd.read_clpt(fn_xi=self.inis['clpt_xi'], fn_v=self.inis['clpt_v'],
                             fn_s=self.inis['clpt_s'])
        self.gsrsd.set_s_mu_sample(self.ss)
        self.gsrsd.set_y_sample()

        self.ndata = len(self.data)  # number of data points

        self.dof = self.ndata - self.npars + 1  # degree of freedom

    def ini_pars(self, nwalkers):
        '''initialize the walkers'''
        # uniform distribution
        pos0 = np.random.random((nwalkers, self.npars))
        for i, p in enumerate(self.pars['pars']):
            width = self.pars[p][2] - self.pars[p][1]
            pos0[:, i] = self.pars[p][1] + width * pos0[:, i]

        return pos0

    def c_model(self, theta):
        '''compute the model vector'''
        for i, p in enumerate(self.pars['pars']):  # map theta to fitting pars
            self.pars[p][0] = theta[i]

        self.gsrsd.set_pars(nu=self.pars['nu'][0], f_v=self.pars['f'][0],
                            sFOG=self.pars['sFOG'][0])
        xi0, xi2 = self.gsrsd.c_xi()
        return np.concatenate((xi0, xi2))

    def ln_prior(self, theta):
        '''ln prior'''
        # uniform in the limited range
        for i, p in enumerate(self.pars['pars']):
            if theta[i] < self.pars[p][1] or theta[i] > self.pars[p][2]:
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
