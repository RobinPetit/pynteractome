from math import factorial
from time import time
import itertools
import graph_tool as gt
import graph_tool.draw as gt_draw
from pynteractome.core.analyses.isomorphism import extract_isomorphism_classes, are_isomorphic
from pynteractome.IO import IO
from pynteractome.utils import sec2date, log

__all__ = ['pathogenic_topology_analysis']

_total_nb_subsets = 0
_current_idx = 0
_start_time = None

def _binom(n, k):
    r'''
    Get the binomial factor :math:`\binom kn` defined as:

    .. math::
        \binom nk := \frac {n!}{k!(n-k)!}
    '''
    return factorial(n) // (factorial(k) * factorial(n-k))

def pathogenic_topology_analysis(integrator, size=2):
    '''
    Perform analysis of common subtopologies within disease modules.

    Args:
        integrator (:class:`LayersIntegrator <pynteractome.layers.LayersIntegrator>`):
            the integrator
        size (int):
            the size of sub-disease-modules to analyse
    '''
    global _total_nb_subsets, _start_time
    try:
        iso_counts = IO.load_topology_analysis(integrator, size)
    except KeyError:
        iso_counts = list()
    idx = 0
    all_dms, all_terms = _extract_disease_modules(integrator, set(integrator.iter_leaves()), size, iso_counts)
    for dm in all_dms:
        _total_nb_subsets += _binom(len(dm), size)
    _start_time = time()
    for idx, (dm, term) in enumerate(zip(all_dms, all_terms)):
        _extract_all_subtopologies(integrator.interactome, iso_counts, dm, term, size)
    print()
    iso_counts.sort(key=lambda e: len(e[1]), reverse=True)
    print([['G{}'.format(i+1), len(iso_counts[i][1]), sum([len(count) for count in iso_counts[i][1].values()])] for i in range(len(iso_counts))])
    IO.save_topology_analysis(integrator, iso_counts, size)

def _extract_all_subtopologies(interactome, iso_counts, dm, term, size):
    r'''
    Extract he subtopologies of a disease module and add them to ``iso_counts``.

    Structure of ``iso_counts``:
        >>> type(iso_counts)
        <class 'list'>
        >>> type(iso_counts[0])
        <class 'list'>
        >>> type(iso_counts[0][0]), type(iso_counts[0][1])
        (<class 'graph_tool.GraphView'>, <class 'dict'>)
        >>> arbitrary_key = list(iso_counts[0][1].keys())[0]
        >>> type(iso_counts[0][1][arbitrary_key])
        <class 'list'>
        >>> type(iso_counts[0][1][arbitrary_key][0])
        <class 'tuple'>

        In other words, if we denote by :math:`N` the size of ``iso_counts``, :math:`T`
        the set of HPO terms, :math:`\mu(t)` the disease module of :math:`t`:

        .. math::
            &\forall 0 \leq i \lneqq N : \forall t \in T : \text{iso_counts}[i][1][t] \subset \mathcal P_{\text{size}}(\mu(t)) \\
            &\qquad\qquad \text{ and } \\
            &\forall \sigma \in \text{iso_counts}[i][1][t] : \Delta_\sigma(\mathcal I) \cong \text{iso_counts}[i][0]

    Args:
        interactome (:class:`Interactome <pynteractome.interactome.interactome.Interactome>`):
            the interactome
        iso_counts (list):
            the container of isomorphisms relations. Structure is detailed in the *Structure* subsection above.
        dm (set):
            set of genes associated to a HPO term
        term (int):
            the HPO term spanning ``dm``
        size (int):
            the size of the subsets to topologically analyse
    '''
    global _current_idx
    subsets = list(itertools.combinations(dm, size))
    if not subsets:
        return
    for subset in subsets:
        _current_idx += 1
        if _current_idx % 250 == 0:
            prop = _current_idx/_total_nb_subsets
            log('{}/{}   ({:3.2f}%)    eta: {}     '.format(_current_idx, _total_nb_subsets, 100*prop, sec2date((time()-_start_time)*(1-prop)/prop)), end='\r')
        subgraph = gt.Graph(interactome.get_subgraph(subset, genes=True), prune=True)
        found = False
        for idx in range(len(iso_counts)):
            graph = iso_counts[idx][0]
            if are_isomorphic(subgraph, graph):
                if term in iso_counts[idx][1]:
                    iso_counts[idx][1][term].append(subset)
                else:
                    iso_counts[idx][1][term] = [subset]
                found = True
                break
        if not found:
            iso_counts.append([gt.Graph(subgraph, prune=True), dict()])

def _extract_disease_modules(integrator, all_terms, size, iso_counts):
    '''
    Get the disease modules (DMs) of terms having a DM bigger than given size and not already present in iso_counts.

    Args:
        integrator (:class:`LayersIntegrator <pynteractome.layers.LayersIntegrator>`):
            the integrator
        all_terms (set):
            the HPO terms to extract the disease modules of
        size (int):
            minimum size of the disease modules
        iso_counts (list):
            see :func:`_extract_all_subtopologies` for details

    Return:
        tuple:
            - **all_dms** (:class:`list`): the disease modules of provided HPO terms
            - **related_terms** (:class:`list`): the terms associated to the returned
              DMs s.t. ``all_dms[i]`` is the disease module of HPO term ``related_terms[i]``
    '''
    already_computed_terms = set()
    for G, d in iso_counts:
        already_computed_terms.update(set(d.keys()))
    print(len(all_terms), len(already_computed_terms), len(all_terms - already_computed_terms))
    all_dms = list()
    related_terms = list()
    all_terms -= already_computed_terms
    for term in all_terms:
        try:
            dm = integrator.get_associated_genes(term)
            if size > len(dm):
                continue
            all_dms.append(dm)
            related_terms.append(term)
        except KeyError as e:
            continue
    return all_dms, related_terms
