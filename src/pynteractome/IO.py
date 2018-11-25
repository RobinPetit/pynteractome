# std
import pickle
import shelve
# ext libs
import numpy as np
# local
from .utils import log, hash_str

__all__ = ['IO']

SHELF_DIR = '/media/robin/DATA/Research/MA-Thesis/'

def _save_to_shelf(d, path):
    with shelve.open(path) as shelf:
        for key in d:
            shelf[key] = d[key]

def _get_from_shelf(key, path):
    with shelve.open(path) as shelf:
        if isinstance(key, (list, tuple)):
            ret = [shelf[k] for k in key]
        else:
            ret = shelf[key]
    return ret

class IO:
    ''' Interface class whose goal is to provide store/fetch instructions
    for the rest of the code. '''

    @staticmethod
    def load_sep(interactome, N, prop_depth=0):
        try:
            return _get_from_shelf(IO.__sep_key(prop_depth), IO.get_shelf_path(interactome))
        except KeyError:
            separations = np.zeros((N, N), dtype=np.float)
            Cs = np.zeros((N, N), dtype=np.float)
            return separations, Cs, 0, 0

    @staticmethod
    def save_sep(interactome, separations, Cs, i, j, prop_depth=0):
        '''
        Save the progression in separation analysis.

        Args:
            interactome (:class:`Interactome <pynteractome.interactome.Interactome>`):
                the interactome
            separations (2D :class:`np.ndarray`):
                matrix of :math:`s_{AB}` separation scores
            Cs (2D :class:`np.ndarray`):
                matrix of :math:`C`-scores
            i (int):
                first index of current progression in Cs and separations
            j (int):
                second index of current progression in Cs and separations
            prop_depth (int):
                depth of propagation of HPO term to genes associations

        Return:
            None
        '''
        data = {
            IO.__sep_key(prop_depth): (separations, Cs, i, j)
        }
        _save_to_shelf(data, IO.get_shelf_path(interactome))

    @staticmethod
    def load_density_cache(interactome):
        log('[Loading Density cache]')
        try:
            return _get_from_shelf('density-cache', IO.get_shelf_path(interactome))
        except KeyError:
            return dict()

    @staticmethod
    def save_density_cache(interactome, cache):
        log('[Saving density cache]')
        _save_to_shelf({'density-cache': cache}, IO.get_shelf_path(interactome))

    @staticmethod
    def load_lcc_cache(interactome):
        log('[Loading lcc_cache]')
        try:
            return _get_from_shelf('lcc-cache', IO.get_shelf_path(interactome))
        except KeyError:
            return dict()

    @staticmethod
    def save_lcc_cache(interactome, cache):
        log('[Saving lcc_cache]')
        _save_to_shelf({'lcc-cache': cache}, IO.get_shelf_path(interactome))

    @staticmethod
    def save_entropy(interactome, hs, H=None):
        ''' Save the isomorphism entropy computed on random subgraphs. The
        entropy of the disease modules is also saved if it is not None.

        Args:
            hs (list):
                all the computed entropy values
            H (float):
                The entropy of the disease modules
        '''
        with shelve.open(IO.get_shelf_path(interactome), 'w') as shelf:
            if H is None and 'H' not in shelf:
                raise ValueError('Entropy of disease modules hasn\'t been computed yet.')
            if 'hs' in shelf.keys():
                hs = shelf['hs'] + hs
            shelf['hs'] = hs
            if H is not None:
                shelf['H'] = H

    @staticmethod
    def load_entropy(interactome):
        return _get_from_shelf(['hs', 'H'], IO.get_shelf_path(interactome))

    @staticmethod
    def get_nb_sims_entropy(interactome):
        try:
            return len(IO.load_entropy(interactome)[0])
        except KeyError:
            return 0

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
        path = SHELF_DIR + IO.hash_interactome_path(interactome) + '.pickle'
        pickle.dump(interactome, open(path, 'wb'))

    @staticmethod
    def hash_interactome_path(interactome):
        return hash_str(interactome.interactome_path)

    @staticmethod
    def get_shelf_path(interactome):
        return SHELF_DIR + IO.hash_interactome_path(interactome) + '.shelf'
