# std
import itertools
from math import factorial
import multiprocessing as mp
from time import time
# ext libs
import graph_tool as gt
import graph_tool.draw as gt_draw
# pynteractome
from pynteractome.core.analyses.isomorphism import extract_isomorphism_classes, are_isomorphic
from pynteractome.IO import IO
from pynteractome.utils import sec2date, log

__all__ = ['pathogenic_topology_analysis']

_total_nb_subsets = 0
_current_idx = mp.Value('i', 0)
_start_time = None

def _binom(n, k):
    r'''
    Get the binomial factor :math:`\binom kn` defined as:

    .. math::
        \binom nk := \frac {n!}{k!(n-k)!}
    '''
    return factorial(n) // (factorial(k) * factorial(n-k))

def pathogenic_topology_analysis(integrator, size=4, n_jobs=1):
    '''
    Perform analysis of common subtopologies within disease modules.

    Args:
        integrator (:class:`LayersIntegrator <pynteractome.layers.LayersIntegrator>`):
            the integrator
        size (int):
            the size of sub-disease-modules to analyse
    '''
    global _total_nb_subsets, _start_time
    log('Pathogenic topology analysis [size={}, n_jobs={}]'.format(size, n_jobs))
    try:
        raise KeyError()  # TODO: remove
        iso_counts = IO.load_topology_analysis(integrator, size)
    except KeyError:
        iso_counts = list()
    idx = 0
    all_dms, all_terms = _extract_disease_modules(integrator, set(integrator.iter_leaves()), size, iso_counts)
    _total_nb_subsets = 0
    _current_idx.value = 0
    for dm in all_dms:
        _total_nb_subsets += _binom(len(dm), size)
    _start_time = time()
    _pathogenic_topology_analysis_parallel(integrator, iso_counts, size, all_dms, all_terms, n_jobs)
    print('\nTotal comp time:', sec2date(time()-_start_time))
    iso_counts.sort(key=lambda e: len(e[1]), reverse=True)
    print([['G{}'.format(i+1), len(iso_counts[i][1]), sum([len(count) for count in iso_counts[i][1].values()])] for i in range(len(iso_counts))])
    IO.save_topology_analysis(integrator, iso_counts, size)

def _pathogenic_topology_analysis_linear(integrator, iso_counts, size, all_dms, all_terms):
    DELTA_T = 3600  #1h
    last_save_time = time()
    log('Going for: {} HPO terms'.format(len(all_dms)))
    for dm, term in zip(all_dms, all_terms):
        _extract_all_subtopologies(integrator.interactome, iso_counts, dm, term, size)
        if time() - last_save_time > DELTA_T:
            log('\n\tSaving iso_counts')
            IO.save_topology_analysis(integrator, iso_counts, size)
            last_save_time = time()
    print()

def _pathogenic_topology_analysis_parallel(integrator, iso_counts, size, all_dms, all_terms, n_jobs=2):
    all_subsets = list()
    for term, dm in zip(all_terms, all_dms):
        all_subsets.extend([(term, x) for x in itertools.combinations(dm, size)])
    log('Going for: {} HPO terms'.format(len(all_dms)))
    I2 = integrator.interactome.copy()
    args = list()
    all_terms_dms = [all_subsets] if n_jobs == 1 else [all_subsets[i::n_jobs] for i in range(n_jobs)]
    for terms_dms in all_terms_dms:
        args.append((I2, terms_dms))
    with mp.Pool(n_jobs) as pool:
        iso_lists = pool.starmap(_extract_all_subtopologies_parallel, args)

def _extract_all_subtopologies(interactome, iso_counts, terms_dms):
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
        terms_dms (list):
            TODO
    '''
    if not terms_dms:
        return
    last_sent_idx = 0
    for idx, (term, subset) in enumerate(terms_dms):
        idx += 1
        if idx % 50 == 0:
            _update_current_idx(idx - last_sent_idx)
            last_sent_idx = idx
        subgraph = gt.Graph(interactome.get_subgraph(subset, genes=True), prune=True)
        found = False
        for i in range(len(iso_counts)):
            graph = iso_counts[i][0]
            if are_isomorphic(subgraph, graph):
                if term in iso_counts[i][1]:
                    iso_counts[i][1][term].append(subset)
                else:
                    iso_counts[i][1][term] = [subset]
                found = True
                break
        if not found:
            iso_counts.append([gt.Graph(subgraph, prune=True), dict()])
    if last_sent_idx < idx:
        _update_current_idx(idx - last_sent_idx)

__last_printed_value = 0

def _update_current_idx(value_to_add):
    global __last_printed_value
    with _current_idx.get_lock():
        _current_idx.value += value_to_add
        idx = _current_idx.value
        if (idx - __last_printed_value) > 500:
            prop = idx/_total_nb_subsets
            log('{:,}/{:,}   ({:3.2f}%)    eta: {}     ' \
                .format(
                    idx, _total_nb_subsets, 100*prop,
                    sec2date((time()-_start_time)*(1-prop)/prop)
                ),
                end='\r'
            )
            __last_printed_value = idx

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

def _extract_all_subtopologies_parallel(interactome, terms_dms):
    iso_counts = list()
    _extract_all_subtopologies(interactome, iso_counts, terms_dms)
    return iso_counts
