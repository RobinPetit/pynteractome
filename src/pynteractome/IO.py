# std
from os.path import isdir
import pickle
import shelve
# ext libs
import numpy as np
# local
from .utils import log, hash_str
from .warning import warning

# TODO: Use os.path.join instead of manual concatenation

__all__ = ['IO']

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

    _SHELF_DIR = None

    @staticmethod
    def set_storage_dir(path):
        if IO._SHELF_DIR is not None:
            warning('(IO.set_storage_dir) --- previous storage [{}] is lost'.format(IO._SHELF_DIR))
        assert isinstance(path, str) and isdir(path)
        IO._SHELF_DIR = path
        if IO._SHELF_DIR[-1] != '/':
            IO._SHELF_DIR += '/'

    @staticmethod
    def get_storage_dir_or_die():
        if IO._SHELF_DIR is None:
            raise ValueError(
                '[IO] Storage dir has not been set yet. ' + \
                'Before asking for any IO operation, ' +
                'set it by calling IO.set_storage_dir')
        return IO._SHELF_DIR

    @staticmethod
    def hash_interactome_path(interactome):
        return hash_str(interactome.interactome_path)

    @staticmethod
    def get_shelf_path(interactome):
        return IO.get_storage_dir_or_die() + IO.hash_interactome_path(interactome) + '.shelf'

    ##### separation analysis

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
    def __sep_key(prop_depth):
        return 'sep-{}prop'.format('' if prop_depth > 0 else 'no-')

    ##### Interactome data

    @staticmethod
    def load_interactome(path, create_if_not_found=True, namecode=None):
        if namecode is not None:
            try:
                return IO.load_interactome_by_namecode(namecode)
            except KeyError:
                pass
        ret = None
        try:
            with open(IO.get_storage_dir_or_die() + hash_str(path) + '.pickle', 'rb') as f:
                ret = pickle.load(f)
        except FileNotFoundError:
            if create_if_not_found:
                from pynteractome.interactome import Interactome
                ret = Interactome(path, namecode=namecode)
                IO.save_interactome(ret)
        return ret

    @staticmethod
    def load_interactome_by_namecode(namecode):
        filename = IO.get_existing_interactomes()[namecode]
        with open(IO.get_storage_dir_or_die() + filename + '.pickle', 'rb') as f:
            ret = pickle.load(f)
        print('Loaded interactome with namecode:', namecode)
        return ret

    @staticmethod
    def save_interactome(interactome):
        path = IO.get_storage_dir_or_die() + IO.hash_interactome_path(interactome) + '.pickle'
        with open(path, 'wb') as f:
            pickle.dump(interactome, f)
        IO._add_existing_interactome(interactome)

    @staticmethod
    def _add_existing_interactome(interactome):
        with open(IO.get_storage_dir_or_die() + '.interactomes', 'a') as f:
            key = interactome.namecode
            filename = IO.hash_interactome_path(interactome)
            f.write(key + ':' + filename + '\n')
            print('Saved interactome as:', interactome.namecode)

    @staticmethod
    def get_existing_interactomes():
        shelf_dir = IO.get_storage_dir_or_die()
        ret = dict()
        try:
            with open(shelf_dir + '.interactomes', 'r') as f:
                for line in f:
                    try:
                        namecode, filename = line.strip().split(':')
                        ret[namecode] = filename
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        return ret
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

    ##### entropy analysis

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

    ##### topology

    @staticmethod
    def save_topology_analysis(integrator, iso_counts, size):
        _save_to_shelf(
            {'topology-{}'.format(size): iso_counts},
            IO.get_shelf_path(integrator.interactome)
        )

    @staticmethod
    def load_topology_analysis(integrator, size):
        return _get_from_shelf(
            'topology-{}'.format(size),
            IO.get_shelf_path(integrator.interactome)
        )
