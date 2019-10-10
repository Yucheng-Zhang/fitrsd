'''Likelihood function.'''
import numpy as np
import collections


class prob:
    '''Model likelihood function class.'''

    def __init__(self, fn):
        '''parse the ini file'''
        self.dic = collections.OrderedDict()
        self.dic['pars'] = ['nu', 'f', 'sFOG']
        self.dic['strs'] = ['xi0file', 'xi2file', 'cov_file',
                            'cosmosel', 'clptdir', 'outroot']

        # load & parse the ini file
        lines = [line.strip() for line in open(fn).readlines()]
        for line in lines:
            ls = line.split()
            # if parameter

        # data vector
        self.data = None

    def c_model(self):
        '''compute the model vector'''
        pass

    def ln_prob(self, theta):
        '''ln probability.'''
        pass
