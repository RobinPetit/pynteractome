# std libs
from time import time
# ext libs
import numpy as np
from graph_tool import GraphView, Graph
from graph_tool.topology import isomorphism
# local
from pynteractome.utils import sec2date, log as _log
from pynteractome.IO import IO

# In order to use Graph[View] in [frozen]sets
# equality and hash functions must be defined properly
Graph.to_tuple = GraphView.to_tuple = lambda s: (tuple(s.vertices()), tuple(s.edges()))
Graph.__eq__ = GraphView.__eq__ = lambda s, v: s.to_tuple() == v.to_tuple()
Graph.__hash__ = GraphView.__hash__ = lambda s: hash(s.to_tuple())


def are_isomorphic(G, H):
    if G.num_vertices() != H.num_vertices():
        return False
    if G.num_edges() != H.num_edges():
        return False
    return G.num_edges() == 0 or isomorphism(G, H)

def extract_isomorphism_classes(graphs):
    if len(graphs) == 1:
        return [graphs]
    graphs.sort(key=lambda graph: graph.num_vertices())
    classes = list()
    indices = np.arange(len(graphs))
    while indices.any():
        _log('Building new isomorphism class {} out of {} remaining    [{:3.2f}%]     ' \
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
    nb_performed_sims = IO.get_nb_sims_entropy()
    nb_sims -= nb_performed_sims
    if nb_sims <= 0:
        _log('{} simulations already performed.'.format(nb_performed_sims))
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
    IO.save_entropy(interactome, entropy_values, H)

def get_entropy_values(nb_sims, nb_vertices, interactome):
    entropy_values = list()
    beg = time()
    for i in range(nb_sims):
        _log('Running simulation {}/{}'.format(i+1, nb_sims), end='')
        if i > 0:
            el_time = time()-beg
            prop = i/nb_sims
            nb_secs = el_time/prop * (1-prop)
            print('\t\teta: {}'.format(sec2date(nb_secs)), end='')
        print('')
        random_subgraphs = [interactome.get_random_subgraph(N) for N in nb_vertices]
        entropy_values.append(isomorphism_entropy(random_subgraphs))
        print('\n')
    return entropy_values

def entropy(S):
    r'''
    Returns $K(|S|)\sum_{s \in S}\frac {|s|}{|T|}\log_2\frac {|s|}{|T|}$,
    where $T = \bigsqcup_{s \in S}s$ is the set of which S is a partition
    and $K(|S|) = \frac 1{\log|S|}$ is the constant normalizing the entropy
    in $[0, 1]$.
    '''
    if len(S) == 1:
        return 0.
    ret = 0.
    len_T = sum(map(len, S))
    for s in S:
        p = len(s)/len_T
        ret += p*np.log(p)
    return -ret/np.log(len(S)), len(S)

def isomorphism_entropy(S):
    return entropy(extract_isomorphism_classes(S))

