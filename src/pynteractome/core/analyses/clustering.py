from pynteractome.utils import log

__all__ = ['clustering_analysis']

def clustering_analysis(integrator, nb_sims):
    _clustering_analysis(integrator, nb_sims, integrator.get_hpo2genes())

def _clustering_analysis(integrator, nb_sims, disease2genes):
    interactome = integrator.interactome
    sizes = set()
    for genes in disease2genes.values():
        genes &= interactome.genes
        if genes:
            sizes.add(len(genes))
    sizes = interactome.where_clustering_cache_nb_sims_lower_than(sizes, nb_sims)
    if sizes:
        log('Filling clustering cache')
        interactome.fill_clustering_cache(nb_sims, sizes)
        log('Clustering cache filled')
    else:
        log('Everything already in cache')
