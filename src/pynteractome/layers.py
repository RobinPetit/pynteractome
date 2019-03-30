# std
import networkx as nx  # damn it obonet
import numpy as np
# local
from .hpo.omim2hpo import get_omim2hpo
from .hpo.load_obo import get_subtree, load_abnormal_phenotypes
from .mendeliome.parser import MendeliomeParser
from .utils import reverse_set_dict, log
from .IO import IO

__integrators = dict()

DEFAULT_PROP_DEPTH = 0

def get_integrator(integrator_config, prop_depth=DEFAULT_PROP_DEPTH):
    if integrator_config.interactome_namecode not in __integrators:
        print('loading interactome')
        interactome = IO.load_interactome(integrator_config.interactome_path,
                                          namecode=integrator_config.interactome_namecode)
        print('loading hpo')
        hpo = load_abnormal_phenotypes()
        print('loading mendeliome')
        mendeliome = MendeliomeParser(
            integrator_config.disease_genes_path,
            integrator_config.disease_names_path,
            integrator_config.gene2omim_path
        )
        __integrators[integrator_config.interactome_namecode] = LayersIntegrator(interactome, mendeliome, hpo, prop_depth)
        print('Got integrator')
    return __integrators[integrator_config.interactome_namecode]

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
            the depth to propagate HPO term :math:`\rightarrow` genes associations (see :meth:`propagate_genes`)

    Attributes:
        interactome (:class:`Interactome <pynteractome.interactome.interactome.Interactome>`):
            the interactome
        mendeliome (:class:`MendeliomeParser <pynteractome.mendeliome.parser.MendeliomeParser>`):
            the mendeliome
        hpo (:class:`nx.Graph`):
            the Human Phenotype Ontology
        propagation_depth (int):
            the current up-propagation depth in the HPO term :math:`\rightarrow` OMIM phenotype associations
        gene2omim (dict):
            mapping Entrez gene :math:`\rightarrow` OMIM phenotypes
        omim2hpo (dict):
            mapping OMIM phenotype :math:`\rightarrow` HPO terms
        gamma_union (dict):
            mapping HPO term :math:`\rightarrow` Entrez gene s.t.:

            .. math::
                \gamma_\cup(t) = \bigcup_{\omega \in \mathcal H_t}\mathcal M^\omega
        gamma_inter (dict):
            mapping HPO term :math:`\rightarrow` Entrez gene s.t.:

            .. math::
                \gamma_\cap(t) = \bigcap_{\omega \in \mathcal H_t}\mathcal M^\omega
        omim2gene (dict):
            mapping OMIM phenotype :math:`\rightarrow` Entrez genes
        hpo2omim (dict):
            mapping HPO term :math:`\rightarrow` OMIM phenotypes
        terms_by_order (list):
            list of sets of HPO terms s.t. terms_by_order[i] is the set of all HPO terms of depth i
        depths (dict):
            mapping HPO term :math:`\rightarrow` depth (depths[term] == term depth)
    '''
    def __init__(self, interactome, mendeliome, hpo, prop_depth=0):
        self.interactome = interactome
        self.hpo = hpo
        self.hpo_root = 118
        self.hpo_terms = set(hpo.nodes())
        self.propagation_depth = 0
        self.mendeliome = mendeliome
        self.gamma_union = self.gamma_inter = None
        self.gamma_union_prime = self.gamma_inter_prime = None
        self.omim2hpo = self.bottom_up_hpo = None

        self._compute_hpo_by_order()
        self._compute_hpo_by_height()
        self._init_mappings(prop_depth)

    ##### Getters

    def get_hpo2genes(self, gene_mapping='intersection'):
        r'''
        Get the mapping HPO terms :math:`\rightarrow` Entrez genes

        Args:
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap` or ``'union'`` to get :math:`\gamma_\cup`

        Return:
            dict:
                the mapping HPO terms :math:`\rightarrow` Entrez genes
        '''
        if gene_mapping == 'intersection':
            return self.gamma_inter
        elif gene_mapping == 'union':
            return self.gamma_union
        else:
            raise ValueError('Hpo2Genes mapping can only be \'intersection\' or \'union\'')

    def get_hpo2omim(self):
        r'''
        Get the mapping HPO terms :math:`\rightarrow` OMIM phenotypes

        Return:
            dict:
                the mapping HPO term :math:`\rightarrow` OMIM phenotypes
        '''
        return self.hpo2omim

    def get_gene2omim(self):
        r'''
        Get the mapping Entrez gene :math:`\rightarrow` OMIM phenotypes

        Return:
            dict:
                the mapping Entrez gene :math:`\rightarrow` OMIM phenotypes
        '''
        return self.gene2omim

    def get_gene2hpo(self, gene_mapping='intersection'):
        r'''
        Get the mapping Entrez genes :math:`\rightarrow` HPO terms

        Args:
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap^{-1} : \mathcal G \to \mathcal P(\mathcal T)` or
                ``'union'`` to get :math:`\gamma_\cup^{-1} : \mathcal G \to \mathcal P(\mathcal T)`

        Return:
            dict:
                the mapping HPO terms :math:`\rightarrow` Entrez genes
        '''
        if gene_mapping == 'intersection':
            return self.gamma_inter_prime
        elif gene_mapping == 'union':
            return self.gamma_union_prime
        else:
            raise ValueError('Genes2Hpo mapping can only be \'intersection\' or \'union\'')

    def het_hpo2omim(self):
        r'''
        Get the mapping OMIM phenotype :math:`\rightarrow` HPO terms

        Return:
            dict:
                the mapping :math:`t \mapsto \mathcal H_t`
        '''

    def get_omim2genes(self):
        r'''
        Get the mapping OMIM phenotypes :math:`\rightarrow` genes

        Return:
            dict:
                the mapping :math:`\omega \mapsto \mathcal M^\omega`
        '''
        return self.omim2gene

    def get_hpo_propagation_depth(self):
        '''
        Get the current propagation depth of the associations.
        '''
        return self.propagation_depth

    def get_hpo_depth(self):
        '''
        Get the maximum depth of a HPO term in the hierarchy.
        '''
        return len(self.terms_by_order)

    def get_term_depth(self, term):
        r'''
        Get the depth (order) of a given term :math:`t` in the hierarchy.

        Args:
            term (int):
                the HPO term

        Return:
            int:
                :math:`\omega(t)`
        '''
        return self.depths[term]

    def get_term_height(self, term):
        r'''
        Get the height of a given term :math:`t` in the hierarchy.

        Args:
            term (int):
                the HPO term

        Return:
            int:
                :math:`\lambda(t)`
        '''
        return self.heights[term]

    def get_disease_modules(self, gene_mapping='intersection'):
        r'''
        Get all the disease modules. See :meth:`get_disease_module`.

        Args:
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap` or ``'union'`` to get :math:`\gamma_\cup`

        Return:
            generator:
                iterable of subgraphs (disease modules)
        '''
        hpo2genes = self.get_hpo2genes(gene_mapping)
        return (self.get_disease_module(term) for term in hpo2genes)

    def get_disease_module(self, term, gene_mapping='intersection'):
        r'''
        Get all the disease module of given HPO term.

        Args:
            term (int):
                HPO term
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap` or ``'union'`` to get :math:`\gamma_\cup`

        Return:
            :class:`graph_tool.GraphView`:
                the disease module of ``term``: :math:`\Delta_{\gamma(t)}(\mathscr I)`
        '''
        return self.interactome.get_subgraph(
            self.get_associated_genes(term, gene_mapping),
            genes=True
        )

    def get_associated_genes(self, term, gene_mapping='intersection'):
        r'''
        Get the genes associated to given HPO term.

        Args:
            term (int):
                HPO term
            gene_mapping (str):
                ``'intersection'`` to get :math:`\gamma_\cap` or ``'union'`` to get :math:`\gamma_\cup`

        Return:
            set:
                the gene set associated to given term
        '''
        return self.get_hpo2genes(gene_mapping)[term] & self.interactome.genes

    ##### Propagation

    def reset_propagation(self):
        if self.get_hpo_propagation_depth() == 0:
            return
        self._init_mappings(0)

    def propagate_genes(self, depth):
        r'''
        TODO: Fix this with new notations
        Up propagate the HPO term :math:`\rightarrow` OMIM phenotype and HPO term :math:`\rightarrow` Entrez genes associations

        Args:
            depth (int):
                the number of individual up-propagation to perform
        '''
        print('Propagating genes')
        if depth < 0:
            depth = self.get_hpo_depth()
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
        self.omim2hpo = reverse_set_dict(self.hpo2omim)
        self._compute_gamma()

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

    def iter_terms(self):
        '''
        Iterate over all the HPO terms from lowest depth to highest depth
        '''
        for n in range(self.get_hpo_depth()):
            for term in sorted(self.order_n_ontology(n)):
                yield term

    def iter_leaves(self, rank_parents=0):
        '''
        Iterate over the leaves in the HP Ontology.

        Args:
            rank_parents (int): the number of generations above the leaves to also consider
        '''
        # TODO: implement parents scanning
        for term in self.hpo.nodes():
            if self.hpo.out_degree(term) == 0:
                yield term

    def get_nb_associated_genes(self, gene_mapping='intersection'):
        r'''
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
        r'''
        Get the sub-interactome induced by taking only the genes of the Mendeliome
        (associated to at least one phenotype)

        Return:
            :class:`Interactome <pynteractome.interactome.interactome.Interactome>`:
                :math:`\Delta_{\mathcal M}(\mathcal I)`
        '''
        genes = self.mendeliome.get_all_genes() & self.interactome.genes
        return self.interactome.get_subinteractome(genes, namecode='mendeliome')

    ##### Private methods

    def _init_mappings(self, prop_depth):
        r'''
        Create mapping representing :math:`\mathcal M`, :math:`\mathcal H`, :math:`\gamma_\cap` and :math:`\gamma_\cup`.
        '''
        self.propagation_depth = 0
        self.gene2omim = {k: v for (k, v) in self.mendeliome.get_gene2omim().items() if k in self.interactome.genes and len(v) > 0}
        self.omim2gene = reverse_set_dict(self.gene2omim)
        omim2hpo = get_omim2hpo()
        self.hpo2omim = {k: v for (k, v) in reverse_set_dict(omim2hpo).items() if k in self.hpo and len(v) > 0}
        self.omim2hpo = reverse_set_dict(self.hpo2omim)
        if prop_depth > 0:
            self.propagate_genes(prop_depth)
        else:
            self._compute_gamma()

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
            self.terms_by_order[idx] = sorted(sub_set)

    def _compute_hpo_by_height(self):
        '''
        Create list such that the ith element is the set of all
        HPO term of height i
        '''
        self.terms_by_height = list()
        self.heights = dict()
        self._get_term_height(self.hpo_root)
        for term, height in self.heights.items():
            while len(self.terms_by_height) <= height:
                self.terms_by_height.append(set())
            self.terms_by_height[height].add(term)
        for idx, sub_set in enumerate(self.terms_by_height):
            self.terms_by_height[idx] = sorted(sub_set)

    def _get_term_height(self, t):
        r'''
        Get the height :math:`\lambda(t)` and store it in `self.heights`
        '''
        if t in self.heights:
            return self.heights[t]
        if len(self.hpo[t]) == 0:
            self.heights[t] = 0
        else:
            self.heights[t] = 1 + min(map(self._get_term_height, self.hpo[t]))
        return self.heights[t]

    def _propagate(self, term, depth):
        if depth == 0 or term not in self.hpo2omim:
            return
        for parent_term in self.bottom_up_hpo.neighbors(term):
            if parent_term not in self.hpo2omim:
                self.hpo2omim[parent_term] = set()
            self.hpo2omim[parent_term] |= self.hpo2omim[term]
            self._propagate(parent_term, depth-1)

    def _compute_gamma(self):
        self.gamma_inter = dict()
        self.gamma_union = dict()
        for term, phenotypes in self.hpo2omim.items():
            if term not in self.hpo:
                continue
            self.gamma_union[term] = set()
            self.gamma_inter[term] = list()
            for phenotype in phenotypes:
                if phenotype in self.omim2gene and len(self.omim2gene[phenotype]) != 0:
                    self.gamma_union[term] |= self.omim2gene[phenotype]
                    self.gamma_inter[term].append(self.omim2gene[phenotype])
        self.gamma_inter = {k: set.intersection(*v) for k, v in self.gamma_inter.items() if v != []}
        self.gamma_inter = {k: v for k, v in self.gamma_inter.items() if len(v) > 0}
        self.gamma_inter_prime = reverse_set_dict(self.gamma_inter)
        self.gamma_union_prime = reverse_set_dict(self.gamma_union)

