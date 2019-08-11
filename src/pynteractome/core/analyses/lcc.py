import numpy as np
import scipy.stats as stats
# local
from pynteractome.utils import log

__all__ = ['lcc_analysis_hpo', 'lcc_analysis_omim', 'normality_test_lcc']

def lcc_analysis_omim(integrator, nb_sims):
    '''
    See :func:`pynteractome.core.analysis.lcc._lcc_analysis` with OMIM phenotypes as diseases.
    '''
    _lcc_analysis(integrator, nb_sims, integrator.get_omim2genes())

def lcc_analysis_hpo(integrator, nb_sims, gene_mapping='intersection'):
    '''
    See :func:`pynteractome.core.analysis.lcc._lcc_analysis` with HPO terms as diseases.
    '''
    print('[[{}]]'.format(gene_mapping))
    integrator.reset_propagation()
    for depth in range(integrator.get_hpo_depth()):
        log('Propagation == {}'.format(depth))
        integrator.propagate_genes(depth)
        assert integrator.get_hpo_propagation_depth() == depth
        _lcc_analysis(integrator, nb_sims, integrator.get_hpo2genes(gene_mapping))
        print('')

def _lcc_analysis(integrator, nb_sims, disease2genes):
    r'''
    Approach the distribution of the LCC size of uniformly sampled disease modules.

    Args:
        integrator (:class:`LayersIntegrator <pynteractome.layers.LayersIntegrator>`):
            the layers integrator
        nb_sims (int):
            minimal number of simulations needed to approximate the probability distribution
        disease2genes (dict):
            mapping from disease :math:`\rightarrow` associated genes. Diseases can be e.g. HPO terms
            or OMIM phenotypes.

    Return:
        None
    '''
    interactome = integrator.interactome
    sizes = set()
    for genes in disease2genes.values():
        genes &= interactome.genes
        if genes:
            sizes.add(len(genes))
    print('{} different sizes have been found'.format(len(sizes)))
    sizes = interactome.where_lcc_cache_nb_sims_lower_than(sizes, nb_sims)
    print('{} sizes are unique to this depth'.format(len(sizes)))
    if sizes:
        log('Filling cache')
        interactome.fill_lcc_cache(nb_sims, sizes)
        log('Cache filled')
    else:
        log('Everything already in cache')

def normality_test_lcc(integrator, p_threshold=.01):
    r'''
    Evaluate whether the LCC size of uniformly sampled disease modules follows a normal distribution.

    Args:
        integrator (:class:`LayersIntegrator <pynteractome.layers.LayersIntegrator>`):
            the layers integrator
        p_threshold (float):
            the statistical threshold. If :math:`p < p_{\text{threshold}}`, then the LCC size
            is considered to not be distributed normally.

    Return:
        dict:
            Mapping HPO term :math:`\rightarrow (p, n, m)` where :math:`p` is
            the :math:`p`-value of Shapiro-Wilk normality test, :math:`m` is the number of genes,
            and :math:`n` is a boolean describing whether the LCC size of uniformly sampled
            disease modules of size :math:`m` are distributed randomly.
    '''
    interactome = integrator.interactome
    cache = interactome.get_lcc_cache()
    shapiro_ps = list()
    genes_count = list()
    ret = dict()
    hpo2genes = integrator.get_hpo2genes()
    for term in integrator.iter_terms():
        if term not in hpo2genes:
            continue
        nb_genes = len(hpo2genes[term] & interactome.genes)
        if nb_genes == 0:
            continue
        lccs = cache[nb_genes]
        shapiro_p = stats.shapiro(lccs)[1]
        genes_count.append(nb_genes)
        shapiro_ps.append(shapiro_p)
        if shapiro_p >= p_threshold:
            print('[Term {:7d}]: {} gene(s)       Shapiro-Wilk p-value: {:.3e}' \
                  .format(term, nb_genes, shapiro_p))
        ret[term] = (shapiro_p, shapiro_p <= p_threshold, nb_genes)
    print('Not normal for {} terms out of {}' \
          .format((np.asarray(shapiro_ps) <= p_threshold).sum(), len(shapiro_ps)))
    return ret
