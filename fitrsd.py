'''Interface for running from command line.'''

import argparse
from code import prob
from code import driver
from code import utils
import sys

if __name__ == "__main__":
    # from command line
    parser = argparse.ArgumentParser(
        description='Fit RSD w/ CLPT GSRSD model.')
    parser.add_argument('-ini_file', type=str, default=None)
    parser.add_argument('-run', type=int, default=0,
                        help='1: run mcmc \n' +
                        '2: continue the run')
    parser.add_argument('-froot', type=str, default='fitrsd',
                        help='root for output files')
    parser.add_argument('-nwalkers', type=int, default=100)
    parser.add_argument('-nsteps', type=int, default=1000,
                        help='number of steps for each walker, including burnin')
    parser.add_argument('-progress', type=int, default=0)

    # get chain related
    parser.add_argument('-get_chain', type=int, default=0,
                        help='get chain from .h5 file')
    parser.add_argument('-burnin', type=int, default=500,
                        help='burnin steps removed')
    args = parser.parse_args()


if __name__ == "__main__":

    print('>> Initializing probability model...')
    # args check
    if args.ini_file == None:
        sys.exit('-ini_file must be specified')

    pm = prob.prob(args.ini_file)  # the probability model
    pm.ini_dm()

    fn_h5 = args.froot + '.h5'

    if args.run == 1:
        driver.run_mcmc(pm, args.nwalkers, args.nsteps,
                        fn_h5, progress=bool(args.progress))
    elif args.run == 2:
        driver.cont_run(pm, fn_h5, args.nsteps, progress=bool(args.progress))
    else:
        pass

    if bool(args.get_chain):
        fo_chain = args.froot + '_chain.dat'
        utils.hdf5_to_txt(fn_h5, fo_chain, pm.pars['pars'], burnin=args.burnin)
    else:
        pass
