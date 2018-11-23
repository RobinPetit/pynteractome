from time import time
import numpy as np
# local
from pynteractome.IO import IO
from pynteractome.utils import log, sec2date, C_score

def _load_d_A_cache(integrator):
    term2genes = integrator.get_hpo2genes()
    log('Precomputing distances')
    d_A_cache = list()
    genes_sets = list()
    for term in integrator.iter_terms():
        if term not in term2genes:
            continue
        genes = term2genes[term] & integrator.interactome.genes
        genes = integrator.interactome.verts_id(genes)
        if genes.any():
            values = integrator.interactome.get_all_dists(genes, genes)
            if values:
                d_A_cache.append(np.mean(values, dtype=np.float32))
                genes_sets.append(genes)
    log('|- Done')
    return d_A_cache, genes_sets

def X(n):
    return n*(n+1)//2

def sep_analysis_menche(integrator):
    d_A_cache, genes_sets = _load_d_A_cache(integrator)
    nb_steps = X(len(genes_sets)-1)
    separations, Cs, i0, j0 = IO.load_sep(
        integrator.interactome, len(genes_sets), integrator.get_hpo_propagation_depth()
    )
    counter = 0
    for i in range(i0):
        counter += len(genes_sets)-1 - i
    counter += j0 - i0
    if counter == nb_steps:  # Everything computed
        log('Everything computed')
        return
    initial_counter = counter
    last_save_time = start_time = time()
    for i in range(i0, len(genes_sets)):
        if i != i0:
            j0 = i
        for j in range(j0+1, len(genes_sets)):
            counter += 1
            d_AB = integrator.interactome.get_d_AB(genes_sets[i], genes_sets[j])
            separations[i, j] = separations[j, i] = float(d_AB - (d_A_cache[i] + d_A_cache[j]) / 2)
            Cs[i, j] = Cs[j, i] = C_score(genes_sets[i], genes_sets[j])
            if counter % 1000 == 0:
                print_sep_proportion(start_time, counter, initial_counter, nb_steps)
                if time() - last_save_time > 1800:
                    IO.save_sep(
                        integrator.interactome, separations, Cs,
                        i, j, integrator.get_hpo_propagation_depth()
                    )
                    last_save_time = time()
    i, j = [len(genes_sets)-1]*2
    IO.save_sep(
        integrator.interactome, separations, Cs,
        i, j, integrator.get_hpo_propagation_depth()
    )

def print_sep_proportion(start_time, counter, initial_counter, nb_steps):
    elapsed = time()-start_time
    proportion = (counter - initial_counter)/(nb_steps - initial_counter)
    nb_secs = elapsed/proportion*(1-proportion)
    log('{} out of {}  ({:.3f}% of remaining and {:.3f}% of total)\teta: {:.2f}s ({})' \
        .format(counter, nb_steps, 100*proportion,
                100*counter/nb_steps, nb_secs, sec2date(int(nb_secs))))
