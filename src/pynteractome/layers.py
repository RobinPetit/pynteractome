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
    r'''
    Integration of the 3 layers that are:

    #. Interactome
    #. OMIM phenotypes
    #. Mendeliome

    Args:
        interactome (:class:`Interactome <pynteractome.interactome.interactome.Interactome>`):
            the interactome
        mendeliome (:class:`MendeliomeParser <pynteractome.mendeliome.parser.MendeliomeParser>`):
            the mendeliome
        hpo (:class:`nx.Graph`):
            the Human Phenotype Ontology
        prop_depth (int):
            the depth to propagate HPO term -> genes associations (see :meth:`propagate_genes`)

    Attributes:
        interactome (:class:`Interactome <pynteractome.interactome.interactome.Interactome>`):
            the interactome
        mendeliome (:class:`MendeliomeParser <pynteractome.mendeliome.parser.MendeliomeParser>`):
            the mendeliome
        hpo (:class:`nx.Graph`):
            the Human Phenotype Ontology
        propagation_depth (int):
            the current up-propagation depth in the HPO term -> OMIM phenotype associations
        alpha (dict):
            mapping Entrez gene -> OMIM phenotypes
        beta (dict):
            mapping OMIM phenotype -> HPO terms
        gamma_union (dict):
            mapping Entrez gene -> HPO terms s.t.:

            .. math::
                \gamma_\cup(g) = \bigcup_{\pi \in \alpha(g)}\beta(\pi)
        gamma_inter (dict):
            mapping Entrez gene -> HPO terms s.t.:

            .. math::
                \gamma_\cap(g) = \bigcap_{\pi \in \alpha(g)}\beta(\pi)
        alpha_prime (dict):
            mapping OMIM phenotype -> Entrez genes (inverse mapping of alpha)
        beta_prime (dict):
            mapping HPO term -> OMIM phenotypes (inverse mapping of beta)
        gamma_union_prime (dict):
            mapping HPO term -> Entrez genes (inverse mapping gamma_union)
        gamma_inter_prime (dict):
            mapping HPO term -> Entrez genes (inverse mapping of gamma_inter)
        terms_by_order (list):
            list of sets of HPO terms s.t. terms_by_order[i] is the set of all HPO terms of depth i
        depths (dict):
            mapping HPO term -> depth (depths[term] == term depth)
    '''
    def __init__(self, interactome, mendeliome, hpo, prop_depth=0):
        self.interactome = interactome
        self.hpo = hpo
        self.propagation_depth = 0
        self.mendeliome = mendeliome
        # gamma_union_prime = \mu_\cup   gamma_inter_prime = \mu_\cap
        self.gamma_union = self.gamma_union_prime = None
        self.gamma_inter = self.gamma_inter_prime = None
        self.alpha = self.alpha_prime = None
        self.beta = self.beta_prime = None

        self._compute_hpo_by_order()
        self._init_mappings(prop_depth)

    def get_hpo2genes(self, gene_mapping='intersection'):
        '''
        Get the mapping HPO terms -> genes

        Args:
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap` or ``'union'`` to get :math:`\gamma_\cup`

        Return:
            dict:
                the mapping HPO terms -> genes
        '''
        if gene_mapping == 'intersection':
            return self.gamma_inter_prime
        elif gene_mapping == 'union':
            return self.gamma_union_prime
        else:
            raise ValueError('Hpo2Genes mapping can only be \'intersection\' or \'union\'')

    def reset_propagation(self):
        if self.get_hpo_propagation_depth() == 0:
            return
        self._init_mappings(0)

    def propagate_genes(self, depth):
        '''
        Up propagate the HPO term -> OMIM phenotype and HPO term -> Entrez genes associations

        Args:
            depth (int):
                the number of individual up-propagation to perform
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
        self._compute_gamma()

    def get_hpo_propagation_depth(self):
        '''
        Get the current propagation depth of the associations.
        '''
        return self.propagation_depth

    def get_sub_interactome_ontology(self, hpo_term):
        '''
        Get the interactome subgraph induced by a HPO sub-hierarchy

        Args:
            hpo_term (int):
                the parent HPO term term

        Return:
            the subinteractome induced by the set of genes that are associated to
            a HPO term that has `hpo_term` as parent somewhere in the hierarchy
        '''
        genes = set()
        for term in get_subtree(self.hpo, hpo_term, parents=False):
            if term in self.gamma_union_prime:
                genes |= self.gamma_union_prime[term]
        return nx.subgraph(self.interactome, genes)

    def order_n_ontology(self, n):
        '''
        Get all the HPO terms of a given depth in the hierarchy.

        Args:
            n (int):
                the depth to fetch

        Return:
            set:
                the set of all HPO terms of demanded depth
        '''
        return self.terms_by_order[n]

    def get_hpo_depth(self):
        '''
        Get the maximum depth of a HPO term in the hierarchy.
        '''
        return len(self.terms_by_order)

    def get_term_depth(self, term):
        '''
        Get the depth of a given term in the hierarchy.

        Args:
            term (int):
                the HPO term

        Return:
            int:
                the depth of `term` in the HPO.
        '''
        return self.depths[term]

    def iter_terms(self):
        '''
        Iterate over all the HPO terms from lowest depth to highest depth
        '''
        for n in range(self.get_hpo_depth()):
            for term in sorted(self.order_n_ontology(n)):
                yield term

    def get_nb_associated_genes(self, gene_mapping='intersection'):
        '''
        Get the number of genes associated to every HPO term.

        Args:
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap` or ``'union'`` to get :math:`\gamma_\cup`

        Return:
            array-like:
                array of number of genes associated to each HPO term
        '''
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
        '''
        Get the sub-interactome induced by taking only the genes of the Mendeliome
        (associated to at least one phenotype)

        Return:
            :class:`Interactome <pynteractome.interactome.interactome.Interactome>`:
                :math:`\Delta_{\mathcal M}(\mathcal I)`
        '''
        genes = self.mendeliome.get_all_genes() & self.interactome.genes
        return self.interactome.get_subinteractome(genes)

    ##### Private methods

    def _init_mappings(self, prop_depth):
        '''
        Creates dictionaries simulating functions alpha, beta, gamma and their
        'prime' versions. See :attr:`alpha`, :attr:`beta`, :attr:`gamma_union`, :attr:`gamma_inter`
        '''
        self.propagation_depth = 0
        # alpha
        self.alpha = self.mendeliome.get_gene2omim()
        self.alpha_prime = self.mendeliome.get_omim2gene()
        # beta
        self.beta = get_omim2hpo()
        self.beta_prime = reverse_set_dict(self.beta)
        self._compute_gamma()
        if prop_depth > 0:
            self.propagate_genes(prop_depth)

    def _compute_hpo_by_order(self):
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

    def _propagate(self, term, depth):
        if depth == 0 or term not in self.beta_prime:
            return
        for parent in self.bottom_up_hpo.neighbors(term):
            if parent not in self.beta_prime:
                self.beta_prime[parent] = set()
            self.beta_prime[parent] |= self.beta_prime[term]
            self._propagate(parent, depth-1)

    def _compute_gamma(self):
        self.gamma_inter = dict()
        self.gamma_union = dict()
        for gene in self.alpha.keys() & self.interactome.genes:
            changed = False
            self.gamma_union[gene] = set()
            self.gamma_inter[gene] = list()
            for phenotype in self.alpha[gene]:
                if phenotype in self.beta:
                    changed = True
                    self.gamma_union[gene] |= self.beta[phenotype]
                    self.gamma_inter[gene].append(self.beta[phenotype])
            if not changed:
                del self.gamma_union[gene]
        self.gamma_inter = {k: set.intersection(*v) for (k, v) in self.gamma_inter.items() if v != []}
        self.gamma_union_prime = reverse_set_dict(self.gamma_union)
        self.gamma_union_prime = {k: v for (k, v) in self.gamma_union_prime.items() if k in self.hpo}
        self.gamma_inter_prime = reverse_set_dict(self.gamma_inter)
        self.gamma_inter_prime = {k: v for (k, v) in self.gamma_inter_prime.items() if k in self.hpo and v != []}

