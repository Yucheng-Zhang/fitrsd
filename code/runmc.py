import emcee
from . import prob
from . import utils


def run_mcmc(pm, nwalkers, burnin, nsteps, froot, progress=False):
    '''main function for runnning MCMC'''
    ndim = pm.npars
    # starting point for each walker
    print('>> Initializing walkers...')
    p0 = pm.ini_pars(nwalkers)

    # set up the backend for hdf5 chain file
    fn_h5 = froot + '.h5'
    backend = emcee.backends.HDFBackend(fn_h5)
    backend.reset(nwalkers, ndim)

    # initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pm.ln_prob,
                                    backend=backend)

    # run MCMC
    print('>> Running MCMC, HDF5 chain file: {0:s}'.format(fn_h5))
    _state = sampler.run_mcmc(p0, burnin+nsteps,
                              store=True, progress=progress)

    # write chain to txt data file
    fo_chain = froot + '_chain.dat'
    utils.hdf5_to_txt(fn_h5, fo_chain, pm.pars['pars'], burnin=burnin)

    # get statistics
    fo_stat = froot + '_stat.dat'
    utils.get_stat(fo_chain, fo_stat, pm.pars['pars'], dof=pm.dof)
