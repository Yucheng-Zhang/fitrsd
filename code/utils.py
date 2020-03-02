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


def jackknife_icov(data, correct=True):
    '''Make inverse covariance matrix w/ jackknife.
    Each row are jks for one variable.'''
    cov = np.cov(data, ddof=0) * (data.shape[1] - 1)
    icov = np.linalg.inv(cov)
    if correct:
        nd = data.shape[0]  # length of the data vector
        ns = data.shape[1]  # number of jackknives
        icov = (1 - (nd+1) / (ns-1)) * icov
    return icov


def hdf5_to_txt(fn, fo, pars, burnin=0):
    '''Convert HDF5 chain to txt data.'''
    reader = emcee.backends.HDFBackend(fn, read_only=True)
    samples = reader.get_chain(discard=burnin, flat=True)
    ln_prob = reader.get_log_prob(discard=burnin, flat=True)

    nwalkers, ndim = reader.shape[0], reader.shape[1]
    burnin *= nwalkers
    nsteps = ln_prob.shape[0]

    data = np.column_stack((ln_prob, samples))

    header = 'FitRSD MCMC chain\n'
    header += 'Raw HDF5 file: {0:s}\n'.format(fn)
    header += 'burnin: {0:d}, nsteps: {1:d}\n'.format(burnin, nsteps)
    header += 'number of fitting parameters: {0:d}\n'.format(ndim)
    header += 'ln_prob'
    for par in pars:
        header = header + ' '*5 + par

    np.savetxt(fo, data, header=header, fmt='%15.8e')


def get_stat(fn, fo, pars, dof=None):
    '''Get statistics from MCMC chain file.'''
    chain = np.loadtxt(fn)
    ibf = np.argmax(chain[:, 0])  # best-fit (max-likelihood) index

    databf = chain[ibf]
    data_mean = np.mean(chain, axis=0)
    data_sigma = np.std(chain, axis=0, ddof=0)

    data = np.array([databf, data_mean, data_sigma])
    data[:, 0] = -2. * data[:, 0]  # chi2

    header = 'row: best-fit, mean, sigma\n'
    if dof != None:
        header += 'dof = {0:d}\n'.format(dof)
    header += 'chi2'
    for par in pars:
        header = header + ' '*5 + par
    fmt = '%.2f' + '   %15.8e' * len(pars)
    np.savetxt(fo, data, header=header, fmt=fmt)
