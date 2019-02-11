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
from pynteractome.isomorphism_counts import IsomorphismCounts
from pynteractome.IO import IO
from pynteractome.utils import sec2date, log

__all__ = ['pathogenic_topology_analysis']

_total_nb_subsets = 0
_current_idx = mp.Value('i', 0)
_start_time = None

def _binom(n, k):
    r'''
    Get the binomial factor :math:`\binom nk` defined as:

    .. math::
        \binom nk := \frac {n!}{k!(n-k)!}
    '''
    return factorial(n) // (factorial(k) * factorial(n-k))

def pathogenic_topology_analysis(integrator, size=4, n_jobs=1):
    r'''
    Perform analysis of common subtopologies within disease modules.

    **Details**

    Consider a graph :math:`\mathscr G` on :math:`\mathcal G` (e.g. :math:`\mathscr I` or :math:`\mathscr M`).
    Let :math:`\Lambda_\alpha` be the set of couples :math:`(t, \sigma)` where :math:`t \in \mathcal T` and
    :math:`\sigma \in \mathcal P_\alpha(\mu(t))`:

    .. math::
        \Lambda_\alpha := \bigcup_{t \in \mathcal T}\left\{(t, \sigma)\right\}_{\sigma \in \mathcal P_\alpha(\mu(t)}.

    Let :math:`\Gamma(\alpha) := \mathcal G(\alpha)/\cong` be the set of non-isomorphic graphs with :math:`\alpha`
    vertices, and define the canonical mapping:

    .. math::
        \Xi_\alpha : \mathcal P_\alpha(\mathcal G) \to \Gamma(\alpha) : \sigma = \{g_i\}_{i=1}^n \mapsto [\Delta_\sigma(\mathscr G)].

    Let :math:`\sim_\alpha` be an equivalence relation onto :math:`\Lambda_\alpha` defined by:

    .. math::
        (t, \sigma) \sim_\alpha (t', \sigma') \; \iff \; \Xi_\alpha(\sigma) \cong \Xi_\alpha(\sigma').

    Finally consider :math:`\Omega_\alpha`, the canonical extension of :math:`\Lambda_\alpha/\sim_\alpha` defined by:

    .. math::
        \Omega_\alpha := \left\{(\Xi_\alpha(\sigma), [(t, \sigma)])\right\}_{[(t, \sigma)] \in \Lambda_\alpha/\sim_\alpha}.

    Args:
        integrator (:class:`LayersIntegrator <pynteractome.layers.LayersIntegrator>`):
            the integrator
        size (int):
            the size of sub-disease-modules to analyse
    '''
    global _total_nb_subsets, _start_time
    log('Pathogenic topology analysis [size={}, n_jobs={}]'.format(size, n_jobs))
    try:
        iso_counts = IO.load_topology_analysis(integrator, size)
    except KeyError:
        log('\tNo records found. Creating new ones')
        iso_counts = IsomorphismCounts()
    idx = 0
    all_dms, all_terms = _extract_disease_modules(integrator, set(integrator.iter_leaves()), size, iso_counts)
    _total_nb_subsets = 0
    _current_idx.value = 0
    for dm in all_dms:
        _total_nb_subsets += _binom(len(dm), size)
    _start_time = time()
    _pathogenic_topology_analysis_parallel(integrator, iso_counts, size, all_dms, all_terms, n_jobs)
    print('\nTotal comp time:', sec2date(time()-_start_time))
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
        iso_counts_list = pool.starmap(_extract_all_subtopologies_parallel, args, chunksize=1)
    for iso_counts_process in iso_counts_list:
        iso_counts.merge(iso_counts_process)

def _extract_all_subtopologies(interactome, iso_counts, terms_dms):
    r'''
    Extract he subtopologies of a disease module and add them to ``iso_counts``.

    Args:
        interactome (:class:`Interactome <pynteractome.interactome.interactome.Interactome>`):
            the interactome
        iso_counts (:class:`IsomorphismCounts <pynteractome.isomorphism_counts.IsomorphismCounts>`):
            the container of isomorphisms relations.
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
        iso_counts.add(subgraph, term, subset)
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
        iso_counts (:class:`IsomorphismCounts <pynteractome.isomorphism_counts.IsomorphismCounts>`):
            TODO

    Return:
        tuple:
            - **all_dms** (:class:`list`): the disease modules of provided HPO terms
            - **related_terms** (:class:`list`): the terms associated to the returned
              DMs s.t. ``all_dms[i]`` is the disease module of HPO term ``related_terms[i]``
    '''
    already_computed_terms = set()
    for G, d in iso_counts.items():
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
    iso_counts = IsomorphismCounts()
    _extract_all_subtopologies(interactome, iso_counts, terms_dms)
    return iso_counts
