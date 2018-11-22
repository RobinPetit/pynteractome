# std
import pickle
import shelve
# ext libs
import numpy as np
# local
from .utils import save_to_shelf, get_from_shelf, SHELF_DIR, log, hash_str

class IO:
    ''' Interface class whose goal is to provide store/fetch instructions
    for the rest of the code. '''

    _ENTROPY_SHELF_PATH = SHELF_DIR + 'entropy.shelf'

    @staticmethod
    def load_sep(N, prop_depth=0):
        try:
            return get_from_shelf(IO.__sep_key(prop_depth))
        except KeyError:
            separations = np.zeros((N, N), dtype=np.float)
            Cs = np.zeros((N, N), dtype=np.float)
            return separations, Cs, 0, 0

    @staticmethod
    def save_sep(separations, Cs, i, j, prop_depth=0):
        '''
        TODO: complete by describing the variables
        '''
        data = {
            IO.__sep_key(prop_depth): (separations, Cs, i, j)
        }
        save_to_shelf(data)

    @staticmethod
    def load_density_cache(path):
        log('[Loading Density cache]')
        try:
            return get_from_shelf('density-cache-'+path)
        except KeyError:
            return dict()

    @staticmethod
    def save_density_cache(path, cache):
        log('[Saving density cache]')
        save_to_shelf({'density-cache-'+path: cache})

    @staticmethod
    def load_lcc_cache(path):
        log('[Loading lcc_cache]')
        try:
            return get_from_shelf('lcc_cache-'+path)
        except KeyError:
            return dict()

    @staticmethod
    def save_lcc_cache(path, cache):
        log('[Saving lcc_cache]')
        save_to_shelf({'lcc_cache-'+path: cache})

    @staticmethod
    def save_entropy(hs, H=None):
        ''' Save the isomorphism entropy computed on random subgraphs. The
        entropy of the disease modules is also saved if it is not None.

        Params:
            hs: list
                all the computed entropy values
            H: float
                The entropy of the disease modules
        '''
        with shelve.open(IO._ENTROPY_SHELF_PATH) as shelf:
            if H is None and 'H' not in shelf:
                raise ValueError('Entropy of disease modules hasn\'t been computed yet.')
            if 'hs' in shelf.keys():
                hs = shelf['hs'] + hs
            shelf['hs'] = hs
            if H is not None:
                shelf['H'] = H

    @staticmethod
    def load_entropy():
        return get_from_shelf(['hs', 'H'], IO._ENTROPY_SHELF_PATH)

    @staticmethod
    def get_nb_sims_entropy():
        return len(IO.load_entropy()[0])

    @staticmethod
    def __sep_key(prop_depth):
        return 'sep-{}prop'.format('' if prop_depth > 0 else 'no-')


    @staticmethod
    def load_interactome(path, create_if_not_found=True):
        ret = None
        try:
            ret = pickle.load(open(SHELF_DIR + hash_str(path) + '.pickle', 'rb'))
        except FileNotFoundError:
            if create_if_not_found:
                from pynteractome.interactome import Interactome
                ret = Interactome(path)
                IO.save_interactome(ret)
        return ret

    @staticmethod
    def save_interactome(interactome):
        pickle.dump(interactome, open(SHELF_DIR + hash_str(interactome.interactome_path) + '.pickle', 'wb'))
