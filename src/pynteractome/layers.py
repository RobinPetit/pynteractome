# std
import networkx as nx  # damn it obonet
import numpy as np
# local
from .hpo.omim2hpo import get_omim2hpo
from .hpo.load_obo import get_subtree, load_abnormal_phenotypes
from .mendeliome.parser import MendeliomeParser
from .utils import reverse_set_dict, log
from .IO import IO

__integrator = None

DEFAULT_PROP_DEPTH = 0

def get_integrator(integrator_config, prop_depth=DEFAULT_PROP_DEPTH):
    global __integrator

    if __integrator is None:
        print('loading interactome')
        interactome = IO.load_interactome(integrator_config.interactome_path)
        print('loading hpo')
        hpo = load_abnormal_phenotypes()
        print('loading mendeliome')
        mendeliome = MendeliomeParser(
            integrator_config.disease_genes_path,
            integrator_config.disease_names_path,
            integrator_config.gene2omim_path
        )
        __integrator = LayersIntegrator(interactome, mendeliome, hpo, prop_depth)
        print('Got integrator')
    return __integrator

class LayersIntegrator:
    def __init__(self, interactome, mendeliome, hpo, prop_depth=0):
        self.interactome = interactome
        self.hpo = hpo
        self.propagation_depth = 0
        self.mendeliome = mendeliome
        # gamma_prime = \mu_\cup   gamma_prime_cap = \mu_\cap
        self.gamma = self.gamma_prime = self.gamma_cap = None
        self.alpha = self.alpha_prime = None
        self.beta = self.beta_prime = None

        self.compute_hpo_by_order()
        self.init_mappings(prop_depth)

    def compute_hpo_by_order(self):
        '''
            Create a list such that the ith element is the set of all
            HPO terms of order i
        '''
        self.terms_by_order = list()
        self.depths = dict()
        spl = nx.shortest_path_length(self.hpo, source=118)
        for term, dist in spl.items():
            while len(self.terms_by_order) <= dist:
                self.terms_by_order.append(set())
            self.terms_by_order[dist].add(term)
            self.depths[term] = dist
        for idx, sub_set in enumerate(self.terms_by_order):
            self.terms_by_order[idx] = list(sorted(sub_set))

    def compute_gamma(self):
        self.gamma_cap = dict()
        self.gamma = dict()
        for gene in self.alpha.keys() & self.interactome.genes:
            changed = False
            self.gamma[gene] = set()
            self.gamma_cap[gene] = list()
            for phenotype in self.alpha[gene]:
                if phenotype in self.beta:
                    changed = True
                    self.gamma[gene] |= self.beta[phenotype]
                    self.gamma_cap[gene].append(self.beta[phenotype])
            if not changed:
                del self.gamma[gene]
        self.gamma_cap = {k: set.intersection(*v) for (k, v) in self.gamma_cap.items() if v != []}
        self.gamma_prime = reverse_set_dict(self.gamma)
        self.gamma_prime = {k: v for (k, v) in self.gamma_prime.items() if k in self.hpo}
        self.gamma_prime_cap = reverse_set_dict(self.gamma_cap)
        self.gamma_prime_cap = {k: v for (k, v) in self.gamma_prime_cap.items() if k in self.hpo and v != []}

    def get_hpo2genes(self, gene_mapping='intersection'):
        if gene_mapping == 'intersection':
            return self.gamma_prime_cap
        elif gene_mapping == 'union':
            return self.gamma_prime
        else:
            raise ValueError('Hpo2Genes mapping can only be \'intersection\' or \'union\'')

    def init_mappings(self, prop_depth):
        '''
            Creates dictionaries simulating functions alpha, beta, gamma and their
            'prime' versions that respectively map:
                + genes -> phenotypes (alpha)
                + phenotypes -> hpo (beta)
                + genes -> hpo (gamma)
                + and their 'prime' versions representing the inverses
        '''
        self.propagation_depth = 0
        # alpha
        self.alpha = self.mendeliome.get_gene2omim()
        self.alpha_prime = self.mendeliome.get_omim2gene()
        # beta
        self.beta = get_omim2hpo()
        self.beta_prime = reverse_set_dict(self.beta)
        self.compute_gamma()
        if prop_depth > 0:
            self.propagate_genes(prop_depth)

    def reset_propagation(self):
        if self.get_hpo_propagation_depth() == 0:
            return
        self.init_mappings(0)

    def propagate_genes(self, depth):
        '''
            Up propagate the HPO term-OMIM phenotype associations
        '''
        new_depth = depth - self.propagation_depth
        if new_depth < 0:
            log('Unable to up-propagate by {} since already propagated of {}' \
                .format(depth, self.propagation_depth))
            return
        self.bottom_up_hpo = self.hpo.reverse()

        for order in reversed(range(self.get_hpo_depth())):
            for term in self.order_n_ontology(order):
                self._propagate(term, depth)
        del self.bottom_up_hpo
        self.propagation_depth = depth
        self.beta = reverse_set_dict(self.beta_prime)
        self.compute_gamma()

    def _propagate(self, term, depth):
        if depth == 0 or term not in self.beta_prime:
            return
        for parent in self.bottom_up_hpo.neighbors(term):
            if parent not in self.beta_prime:
                self.beta_prime[parent] = set()
            self.beta_prime[parent] |= self.beta_prime[term]
            self._propagate(parent, depth-1)

    def get_hpo_propagation_depth(self):
        return self.propagation_depth

    def get_sub_interactome_ontology(self, hpo_term):
        '''
            Get the interactome subgraph induced by a HPO hierarchy
        '''
        genes = set()
        for term in get_subtree(self.hpo, hpo_term, parents=False):
            if term in self.gamma_prime:
                genes |= self.gamma_prime[term]
        return nx.subgraph(self.interactome, genes)

    def order_n_ontology(self, n):
        return self.terms_by_order[n]

    def get_hpo_depth(self):
        return len(self.terms_by_order)

    def get_term_depth(self, term):
        return self.depths[term]

    def iter_terms(self):
        for n in range(self.get_hpo_depth()):
            for term in sorted(self.order_n_ontology(n)):
                yield term

    def get_nb_associated_genes(self, gene_mapping='intersection'):
        term2genes = self.get_hpo2genes(gene_mapping)
        nb_genes = list()
        for term in self.iter_terms():
            if term not in term2genes:
                continue
            genes = term2genes[term] & self.interactome.genes
            genes = self.interactome.verts_id(genes)
            N = len(genes)
            if N == 0:
                continue
            values = self.interactome.get_all_dists(genes, genes)
            if values:
                nb_genes.append(N)
        return np.asarray(nb_genes, dtype=np.int)

    def extract_mendeliome_from_interactome(self):
        genes = self.mendeliome.get_all_genes() & self.interactome.genes
        return self.interactome.get_subinteractome(genes)
