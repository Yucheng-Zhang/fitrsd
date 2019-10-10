import sys


def get_cosmology(tag):
    '''List of cosmology parameters.'''
    if tag == 'planck18':
        cosmo = {'H0': 67.66, 'As': 2.105e-9, 'ns': 0.9665, 'Om0': 0.3111,
                 'Ombh2': 0.02242, 'Omch2': 0.11933, 'Omk': 0, 'mnu': 0}

    else:
        sys.exit('!! exit: wrong cosmology tag !!')

    cosmo['h2'] = (cosmo['H0'] / 100.)**2
    return cosmo
