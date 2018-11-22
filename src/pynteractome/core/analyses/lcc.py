import numpy as np
import scipy.stats as stats
# local
from pynteractome.utils import log

def lcc_analysis_omim(integrator, nb_sims):
    _lcc_analysis(integrator, nb_sims, integrator.alpha_prime)

def lcc_analysis_hpo(integrator, nb_sims):
    for depth in range(integrator.get_hpo_depth()):
        log('Propagation == {}'.format(depth))
        integrator.propagate_genes(depth)
        _lcc_analysis(integrator, nb_sims, integrator.get_hpo2genes())
        print('')

def _lcc_analysis(integrator, nb_sims, disease2genes):
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
    interactome = integrator.interactome
    cache = interactome.get_lcc_cache()
    shapiro_ps = list()
    genes_count = list()
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
    print('Not normal for {} terms out of {}' \
          .format((np.asarray(shapiro_ps) <= p_threshold).sum(), len(shapiro_ps)))
