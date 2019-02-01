# std libs
from time import time
from multiprocessing import Pool, Value
# ext libs
import numpy as np
from graph_tool import GraphView, Graph
from graph_tool.topology import isomorphism
# local
from pynteractome.utils import sec2date, log
from pynteractome.IO import IO

__all__ = ['isomorphism_entropy_analysis', 'extract_isomorphism_classes', 'are_isomorphic']

# In order to use Graph[View] in [frozen]sets
# equality and hash functions must be defined properly
Graph.to_tuple = GraphView.to_tuple = lambda s: (tuple(s.vertices()), tuple(s.edges()))
Graph.__eq__ = GraphView.__eq__ = lambda s, v: s.to_tuple() == v.to_tuple()
Graph.__hash__ = GraphView.__hash__ = lambda s: hash(s.to_tuple())


def are_isomorphic(G, H):
    r'''
    Determine if two graphs G and H are isomorphic to one another.

    Args:
        G (graph_tool.Graph): first graph
        H (graph_tool.Graph): second graph

    Return:
        True if :math:`G \cong H`, False otherwise
    '''
    if G.num_vertices() != H.num_vertices():
        return False
    if G.num_edges() != H.num_edges():
        return False
    return G.num_edges() == 0 or isomorphism(G, H)

def extract_isomorphism_classes(graphs, verbose=False):
    r'''
    Extract the isomorphism classes among the provided graphs.

    Args:
        graphs (list):
            list of :class:`graph_tool.Graph` instances among which the
            isomorphisms are looked at.

    Return:
        list:
            a list ``L`` of list such that if :math:`N` is ``len(L)``, then for :math:`0 \leq i \leq N`:
            :math:`\forall G, H \in L[i] : G \cong H`.
    '''
    if len(graphs) == 1:
        return [graphs]
    graphs.sort(key=lambda graph: graph.num_vertices())
    classes = list()
    indices = np.arange(len(graphs))
    while indices.any():
        if verbose:
            log('Building new isomorphism class {} out of {} remaining    [{:3.2f}%]     ' \
                .format(len(indices), len(graphs), 100*(1 - len(indices)/len(graphs))),
                end='\r')
        to_remove = [0]
        G = graphs[indices[0]]
        classes.append([G])
        for j in range(1, len(indices)):
            H = graphs[indices[j]]
            if H.num_vertices() > G.num_vertices():
                break
            if are_isomorphic(G, H):
                classes[-1].append(graphs[indices[j]])
                to_remove.append(j)
        indices = np.delete(indices, to_remove)
    return classes

def isomorphism_entropy_analysis(integrator, nb_sims):
    nb_performed_sims = IO.get_nb_sims_entropy(integrator.interactome)
    nb_sims -= nb_performed_sims
    if nb_sims <= 0:
        log('{} simulations already performed.'.format(nb_performed_sims))
        return
    if integrator.get_hpo_propagation_depth() != 0:
        integrator.reset_propagation()
    interactome = integrator.interactome
    nb_vertices = list()
    disease_modules = list()
    disease_modules_genes = list()
    for hpo_term in sorted(integrator.get_hpo2genes().keys()):
        genes = integrator.get_hpo2genes()[hpo_term] & interactome.genes
        if len(genes) > 1:
            nb_vertices.append(len(genes))
            genes = np.asarray(interactome.verts_id(genes))
            disease_modules_genes.append(genes)
    entropy_values = get_entropy_values(nb_sims, nb_vertices, interactome)
    H = None
    if nb_performed_sims == 0:
        for genes in disease_modules_genes:
            disease_modules.append(interactome.get_subgraph(genes))
        H = isomorphism_entropy(disease_modules)
    print('')
    IO.save_entropy(interactome, entropy_values, H)

_idx = Value('i', 0)
_beg = 0
_nb_sims = 0

def get_entropy_values(nb_sims, nb_vertices, interactome, n_jobs=4, verbose=False):
    global _beg, _nb_sims
    assert n_jobs > 0
    _beg = time()
    _nb_sims = nb_sims
    _idx.value = 0
    if n_jobs == 1:
        entropy_values = list()
        for i in range(nb_sims):
            entropy_values.append(_process_entropy(interactome, nb_vertices, nb_sims, verbose))
    else:
        I2 = interactome.copy()
        args = [(I2, nb_vertices)] * nb_sims
        with Pool(n_jobs) as pool:
            entropy_values = pool.starmap(_process_entropy, args)
    return entropy_values

def _process_entropy(interactome, nb_vertices):
    with _idx.get_lock():
        _idx.value += 1
        n = _idx.value
    if n % 50 == 0:
        el_time = time()-_beg
        prop = n/_nb_sims
        nb_secs = el_time/prop * (1-prop)
        eta_str = '\t\teta: {}'.format(sec2date(nb_secs))
        log('Running simulation {:3.2f}% {}'.format(100*prop, eta_str), end='')
    random_subgraphs = [interactome.get_random_subgraph(N) for N in nb_vertices]
    ret = isomorphism_entropy(random_subgraphs, False)
    print('', end='\r')
    return ret

def entropy(S):
    r'''
    Compute the entropy induced by the isomorphism classes.

    Args:
        S (list): a list of isomorphism classes

    Return:
        tuple:
            :math:`(H(S), N)` where :math:`N` is the number of isomorphism classes and:

            .. math::
                H(S) = -K(|S|)\sum_{s \in S}\frac {|s|}{|T|}\log_2\frac {|s|}{|T|},

            where :math:`T = \bigsqcup_{s \in S}s` is the set of which S is a partition
            and :math:`K(|S|) = \frac 1{\log|T|}` is the constant normalizing the entropy
            in :math:`[0, 1]`.

            Note the following convention: :math:`H(\emptyset) := 0`.
    '''
    if len(S) == 1:
        return 0.
    ret = 0.
    len_T = sum(map(len, S))
    for s in S:
        p = len(s)/len_T
        ret += p*np.log(p)
    return -ret/np.log(len_T), len(S)

def isomorphism_entropy(S, verbose=False):
    r'''
    Compute the entropy H(S).

    Args:
        S (list): list of :class:`graph_tool.Graph` instances

    Return:
        tuple:
            :math:`(H(S/\sim), |S/\sim|)` where :math:`\sim` is the isomorphism equivalence relation.
    '''
    return entropy(extract_isomorphism_classes(S, verbose))

