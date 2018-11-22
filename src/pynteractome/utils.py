from datetime import datetime as dt
from hashlib import md5
import shelve
import numpy as np

SHELF_DIR = '/media/robin/DATA/Research/MA-Thesis/'
SHELF_PATH = SHELF_DIR + 'data.shelf'

def fmt_g(value):
    return '{:g}'.format(value)

def fmt_e(value):
    return '{:.0e}'.format(value)

def sec2date(secs):
    if not isinstance(secs, int):
        secs = int(secs)
    return '{}d {:2d}h {:2d}m {:2d}s' \
           .format(secs//86400, (secs//3600) % 24, (secs // 60) % 60, secs % 60)

def reverse_set_dict(d):
    ret = dict()
    for key in d:
        for value in d[key]:
            if value not in ret:
                ret[value] = {key}
            else:
                ret[value].add(key)
    return ret

def now():
    return dt.now().strftime('%Y-%m-%d %T.%f')

def log(txt, end='\n'):
    print('[{}]\t{}'.format(now(), txt), end=end, flush=True)

def C_score(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return len(A & B) / min(len(A), len(B))

def J_score(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return len(A & B) / len(A | B)

def save_to_shelf(d, path=SHELF_PATH):
    with shelve.open(path) as shelf:
        for key in d:
            shelf[key] = d[key]

def get_from_shelf(key, path=SHELF_PATH):
    with shelve.open(path) as shelf:
        if isinstance(key, (list, tuple)):
            ret = [shelf[k] for k in key]
        else:
            ret = shelf[key]
    return ret

def extract_triangular(array, indices):
    ret = list()
    for i, index in enumerate(indices):
        if not index:
            continue
        for j in range(i+1, len(indices)):
            if indices[j]:
                ret.append(array[i, j])
    assert len(ret) == indices.sum()*(indices.sum()-1)//2
    return np.asarray(ret, dtype=array.dtype)

def hash_str(s):
    return md5(s.encode('utf-8')).hexdigest()
