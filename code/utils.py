import sys
import numpy as np
import emcee


def get_cosmology(tag):
    '''List of cosmology parameters.'''
    if tag == 'planck18':
        cosmo = {'H0': 67.66, 'As': 2.105e-9, 'ns': 0.9665, 'Om0': 0.3111,
                 'Ombh2': 0.02242, 'Omch2': 0.11933, 'Omk': 0, 'mnu': 0}

    else:
        sys.exit('!! exit: wrong cosmology tag !!')

    cosmo['h2'] = (cosmo['H0'] / 100.)**2
    return cosmo


def jackknife_cov(data):
    '''Make covariance matrix w/ jackknife.
    Each row are jks for one variable.'''
    cov = np.cov(data, ddof=0) * (data.shape[1] - 1)
    return cov


def hdf5_to_txt(fn, fo, burnin=0):
    '''Convert HDF5 chain to txt data.'''
    reader = emcee.backends.HDFBackend(fn, read_only=True)
    samples = reader.get_chain(discard=burnin, flat=True)

    header = 'FitRSD MCMC chain\n'
    header += 'Raw HDF5 file: {0:s}\n'.format(fn)
    header += 'nu   f   sFOG'
    np.savetxt(fo, samples, header=header)
