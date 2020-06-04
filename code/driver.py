import emcee
from . import prob
from . import utils


def run_mcmc(pm, nwalkers, nsteps, fn_h5, progress=False):
    '''Main function for runnning MCMC.'''
    ndim = pm.npars
    # starting point for each walker
    print('>> Initializing walkers...')
    p0 = pm.ini_pars(nwalkers)

    # set up the backend for hdf5 chain file
    backend = emcee.backends.HDFBackend(fn_h5)
    backend.reset(nwalkers, ndim)

    # initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pm.ln_prob,
                                    backend=backend)

    # run MCMC
    print('>> Running MCMC, HDF5 chain file: {:s}'.format(fn_h5))
    _state = sampler.run_mcmc(p0, nsteps, progress=progress)


def cont_run(pm, fn_chain, nsteps, progress=False):
    '''Continue the mcmc run w/ a chain.
    nwalkers is determined from the chain.'''
    # load the chain
    backend = emcee.backends.HDFBackend(fn_chain)
    nwalkers, ndim = backend.shape[0], backend.shape[1]
    print('>> Info of the chain <<')
    print('>> chain file: {:s}'.format(fn_chain))
    print('>> Number of walkers: {:d}'.format(nwalkers))
    print('>> Number of parameters: {:d}'.format(ndim))
    print('>> Iterations: {:d}'.format(backend.iteration))

    # simple check on the num. of parameters
    if pm.npars != ndim:
        raise ValueError('# pars in config: {:d}'.format(pm.npars))

    # continue the run
    print('>> Running MCMC ...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pm.ln_prob,
                                    backend=backend)
    _state = sampler.run_mcmc(None, nsteps, progress=True)
