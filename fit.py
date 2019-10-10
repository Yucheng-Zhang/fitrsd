'''Run MCMC to do the fitting.'''
import emcee
import argparse

if __name__ == "__main__":
    # from command line
    parser = argparse.ArgumentParser(
        description='Fit RSD w/ CLPT GSRSD model.')
    parser.add_argument('-ini_file', type=str, default=None)
    args = parser.parse_args()


def run_mcmc():
    pass


if __name__ == "__main__":
    pass
