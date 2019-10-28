'''Interface for running from command line.'''

import argparse
from code import prob
from code import driver
import sys

if __name__ == "__main__":
    # from command line
    parser = argparse.ArgumentParser(
        description='Fit RSD w/ CLPT GSRSD model.')
    parser.add_argument('-ini_file', type=str, default=None)
    parser.add_argument('-nwalkers', type=int, default=10)
    parser.add_argument('-froot', type=str, default='fitrsd')
    parser.add_argument('-burnin', type=int, default=10,
                        help='burnin steps, which will not be saved to chain file')
    parser.add_argument('-nsteps', type=int, default=10,
                        help='number of steps for each walker')
    parser.add_argument('-progress', type=int, default=0)
    parser.add_argument('-run_mcmc', type=int, default=1)
    args = parser.parse_args()


if __name__ == "__main__":

    print('>> Initializing probability model...')
    # args check
    if args.ini_file == None:
        sys.exit('-ini_file must be specified')

    pm = prob.prob(args.ini_file)  # the probability model

    if bool(args.run_mcmc):
        driver.run_mcmc(pm, args.nwalkers, args.burnin, args.nsteps,
                        args.froot, progress=bool(args.progress))
    else:
        driver.run_stat(pm, args.froot, args.burnin)
