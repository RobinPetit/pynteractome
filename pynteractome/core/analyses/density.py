from pynteractome.utils import log

def density_analysis(integrator, nb_sims):
    return _density_analysis(integrator, nb_sims, integrator.get_hpo2genes())

def _density_analysis(integrator, nb_sims, disease2genes):
    interactome = integrator.interactome
    sizes = set()
    for genes in disease2genes.values():
        genes &= interactome.genes
        if genes:
            sizes.add(len(genes))
    sizes = interactome.where_density_cache_nb_sims_lower_than(sizes, nb_sims)
    if sizes:
        log('Filling density cache')
        interactome.fill_density_cache(nb_sims, sizes)
        log('Density cache filled')
    else:
        log('Everything already in density cache')

