# std
import csv
from copy import deepcopy
from hashlib import md5
from time import time
# ext libs
import numpy as np
from scipy import stats
from graph_tool import Graph, GraphView
from graph_tool.topology import shortest_distance, label_largest_component
from graph_tool.clustering import global_clustering
# local
from pynteractome.utils import sec2date, log
from pynteractome.IO import IO

def X(n):
    '''Compute binomial coefficient n choose 2'''
    return n*(n+1)//2

def _get_lcc_size(G):
    '''Return the size of the largest connected component (LCC) within G.'''
    return label_largest_component(G).a.sum()

def _get_density(G):
    '''Return the density of graph G.'''
    return G.num_edges()  / X(G.num_vertices())

def _get_clustering_coefficient(G):
    '''Return the clustering coefficient :math:`C(G)`.'''
    return global_clustering(G)[0]

class Interactome:
    r'''
    Attributes:
        interactome_path (str):
            the path to the tsv file containing the interactome per se
        namecode (str):
            the name used to recover the (sub)interactome later
        G (:class:`graph_tool.Graph`):
            the internal representation of the interactome as a graph
        genes2vertices (dict):
            mapping Entrez gene :math:`\rightarrow` set of vertices in ``self.G``
        genes (set):
            set of Entrez names of genes present in ``self.G``
        lcc_cache (dict):
            mapping a number of genes to the LCC size of the uniformly sampled subgraphs of this size
        density_cache (dict):
            mapping a number of genes to the density of the uniformly sampled subgraphs of this size
        clustering_cache (dict):
            mapping a number of genes to the clustering coefficient of the uniformly sampled subgraphs of this size
        distances (2D :class:`np.ndarray`):
            matrix of shortest paths from gene :math:`i` to gene :math:`j`
    '''
    def __init__(self, path, namecode=None):
        self.interactome_path = path
        self.namecode = namecode
        self.distances = None
        log('Loading interactome')
        if path is not None:
            self.load_network(path)
        log('interactome loaded')
        self.lcc_cache = self.density_cache = self.clustering_cache = None

    def get_gene_degree(self, gene):
        '''
        Get the degree of a given gene within the interactome.

        Args:
            gene (int): Entrez ID of the gene

        Return:
            int:
                `None` if the gene is not in :math:`\mathscr I` else the number of associated genes within the interactome
        '''
        if gene not in self.genes:
            return None
        vert_id = self.vert_id(gene)
        return self.G.vertex(vert_id).out_degree()

    def set_namecode(self, namecode):
        assert isinstance(namecode, str)
        self.namecode = namecode

    def get_lcc_cache(self):
        '''
        Return the cache of LCC sizes. WARNING: no copy is made.
        Modifying the returned cache can result in undefined behaviour.
        '''
        self.load_lcc_cache()
        return self.lcc_cache

    def load_lcc_cache(self):
        '''Load the cache of LCC sizes simulations if exists, else creates an empty one.'''
        if self.lcc_cache is None:
            self.lcc_cache = IO.load_lcc_cache(self)

    def get_density_cache(self):
        '''
        Return the cache of density. WARNING: no copy is made.
        Modifying the returned cache can result in undefined behaviour.
        '''
        self.load_density_cache()
        return self.density_cache

    def load_density_cache(self):
        '''Load the cache of density simulations if exists, else creates an empty one.'''
        if self.density_cache is None:
            self.density_cache = IO.load_density_cache(self)

    def get_clustering_cache(self):
        '''
        Return the cache of clustering coefficients. WARNING: no copy is made.
        Modifying the returned cache can result in undefined behaviour.
        '''
        self.load_clustering_cache()
        return self.clustering_cache

    def load_clustering_cache(self):
        '''Load the cache of clustering coefficient simulations if exists, else creates an empty one.'''
        if self.clustering_cache is None:
            self.clustering_cache = IO.load_clustering_cache(self)

    def load_network(self, path):
        '''
        Load the interactome stored in a tsv file

        Args:
            path: the path of the interactome file
        '''
        self.G = Graph(directed=False)
        self.genes2vertices = dict()
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            for genes in reader:
                gene1, gene2 = map(int, genes)
                self.add_vertex(gene1)
                self.add_vertex(gene2)
                self.G.add_edge(self.vert_id(gene1), self.vert_id(gene2))
        self.genes = set(self.genes2vertices.keys())
        self.vertices2genes = {v: g for g, v in self.genes2vertices.items()}
        self.compute_spls()

    def add_vertex(self, gene):
        '''
        Create new vertex for `gene` in the graph if not yet present

        Args:
            gene: the name of the gene to ad in the interactome
        '''
        if gene not in self.genes2vertices:
            self.genes2vertices[gene] = len(self.genes2vertices)
            self.G.add_vertex()

    def vert_id(self, gene):
        '''
        Return the id of the desired gene

        Args:
            gene: the gene to retrieve

        Returns:
            the id of the desired gene

        Raises:
            KeyError: if no such gene is in the interactome
        '''
        return self.genes2vertices[gene]

    def verts_id(self, genes, gene_to_ignore=None):
        '''
        Return a list of Vertex instances of the desired genes

        Args:
            genes: an iterable of desired genes
            gene_to_ignore: gene in `genes` that is not desired

        Returns:
            a list of Vertex instances of the desired genes

        Raises:
            KeyError: if any of the genes is not in the interactome
        '''
        return np.array([self.vert_id(gene) for gene in genes if gene != gene_to_ignore])

    def compute_spls(self):
        '''Compute the shortest path between each pair of genes.'''
        if self.distances is not None:
            return
        dists = shortest_distance(self.G)
        self.distances = np.empty((self.G.num_vertices(), self.G.num_vertices()), dtype=np.int)
        for idx, array in enumerate(dists):
            self.distances[idx, :] = array.a[:]

    def get_all_dists(self, A, B):
        '''
        Get a list containing all the distances from a gene in A to the gene set B

        Args:
            A: a source gene set
            B: a destination gene set

        Returns:
            a list of distances [d(a, B) s.t. a in A]
        '''
        insert_self = A is B
        all_dists = list()
        for gene1 in A:
            if insert_self:
                for idx, el in enumerate(B):
                    if el == gene1:
                        indices = np.delete(B, idx)
                        break
            else:
                indices = B
            if not indices.any():
                continue
            indices = np.asarray(indices)
            self.compute_spls()
            dists = self.distances[gene1, indices]
            min_dist = np.min(dists)
            if min_dist > self.G.num_vertices():  # if gene is isolated
                continue                          # go to next gene
            all_dists.append(min_dist)
        return all_dists

    def get_d_A(self, A):
        '''
        Return the inner distance of the disease module A as defined in [1].

        Args:
            A: a gene set

        Returns:
            :math:`d_A`

        References
        ----------

        [1] J. Menche et al., Science 347 , 1257601 (2015). DOI: 10.1126/science.1257601 http://science.sciencemag.org/content/347/6224/1257601
        '''
        return np.mean(self.get_all_dists(A, A))

    def get_d_AB(self, A, B):
        '''
        Return the graph-based distance between A and B as defined in [1].

        Args:
            A: a gene set
            B: a gene set

        Returns:
            :math:`d_{AB}`

        References
        ----------

        [1] J. Menche et al., Science 347 , 1257601 (2015). DOI: 10.1126/science.1257601 http://science.sciencemag.org/content/347/6224/1257601
        '''
        values = self.get_all_dists(A, B)
        values.extend(self.get_all_dists(B, A))
        return np.mean(values, dtype=np.float32)

    def get_random_subgraph(self, size):
        '''
        Uniformly sample a subgraph of given size.

        Args:
            size: number of genes to sample

        Returns:
            A subgraph of self of given size
        '''
        seeds = np.random.choice(len(self.genes), size=size, replace=False)
        return self.get_subgraph(seeds)

    def get_subgraph(self, vertices, genes=False):
        r'''
        Return the subgraph of self induced by the given vertices.

        Args:
            vertices: a set of vertex IDs (or a set of genes)
            genes: a boolean with value `True` if `vertices` is a set of genes
                and `False` if it is a set of vertex IDs.

        Returns:
            :math:`\Delta_{\text{vertices}}(G)`
        '''
        if genes:
            vertices = self.verts_id(vertices)
        filt = self.G.new_vertex_property('bool')
        filt.a[vertices] = True
        return GraphView(self.G, vfilt=filt)

    def get_genes_lcc_size(self, genes):
        r'''
        Return the LCC size of the graph induced by given genes.

        Args:
            genes: an iterable containing genes

        Returns:
            :math:`|LCC(\Delta_{\text{genes}}(G))|`
        '''
        return _get_lcc_size(self.get_subgraph(np.asarray(genes)))

    def get_random_genes_lcc(self, size):
        r'''
        Return the LCC size of a random subgraph of given size.

        Args:
            size (in): number of genes to sample

        Returns:
            :math:`|LCC(\mathcal G(\text{size}, G))|`
        '''
        return _get_lcc_size(self.get_random_subgraph(size))

    def get_random_genes_density(self, size):
        r'''
        Return the density of a random subgraph of given size.

        Args:
            size (int): number of genes to sample

        Returns:
            :math:`d(\mathcal G(\text{size}, G))`
        '''
        return _get_density(self.get_random_subgraph(size))

    def get_genes_density(self, genes):
        r'''
        Return the density of the subgraph induced by given genes.

        Args:
            genes: an iterable of genes

        Returns:
            :math:`d(\Delta_{\text{genes}}(G))`
        '''
        return _get_density(self.get_subgraph(np.asarray(genes)))

    def get_random_genes_clustering(self, size):
        r'''
        Return the clustering coefficient of a random subgraph of given size.

        Args:
            size (int): number of genes to sample

        Returns:
            :math:`C(\mathcal G(\text{size}, G))`
        '''
        G = self.get_random_subgraph(size)
        ret = _get_clustering_coefficient(G)
        return ret

    def get_genes_clustering(self, genes, entrez=False):
        r'''
        Return the clustering coefficient of the subgraph induced by given genes.

        Args:
            genes: an iterable of genes

        Returns:
            :math:`C(\Delta_{\text{genes}}(G))`
        '''
        if entrez:
            genes = self.verts_id(genes)
        return _get_clustering_coefficient(self.get_subgraph(np.asarray(genes)))

    def get_lcc_score(self, genes, nb_sims, shapiro=False, shapiro_threshold=.05):
        r'''
        Get the z-score and the empirical p-value of the LCC size of given genes.

        Args:
            genes (set): gene set
            nb_sims (int): minimum number of simulations for probability distribution estimation
            shapiro (bool): True if normality test is needed, False otherwise (default False)
            shapiro_threshold (float): statistical threshold for normality test

        Returns:
            tuple:
                :math:`(z, p_e, N)` if shapiro is True and :math:`(z, p_e)` otherwise;
                where z is the z-score of the LCC size, :math:`p_e` is the associated
                empirical p-value and N is True if Shapiro-Wilk normality test
                p-value >= shapiro_threshold and False otherwise

        Raises:
            ValueError: if not enough simulations have been performed
        '''
        genes = genes & self.genes
        genes = self.verts_id(genes)
        nb_seeds = len(genes)
        if nb_seeds == 0:
            print('\n\t[Warning: get_lcc_score found no matching gene]')
            return None
        genes_lcc = self.get_genes_lcc_size(genes)
        try:
            lccs = self.get_lcc_cache()[nb_seeds]
            assert len(lccs) >= nb_sims
        except AssertionError:
            raise ValueError(('Only {} simulations found. Expected >= {}. ' + \
                              'fill_lcc_cache has not been called properly') \
                             .format(len(lccs), nb_sims))
        std = lccs.std()
        mean = lccs.mean()
        z = None if std == 0 else float((genes_lcc - mean) / std)
        empirical_p = (lccs >= genes_lcc).sum() / len(lccs)
        if shapiro:
            is_normal = stats.shapiro(np.random.choice(lccs, size=5000))[1] >= shapiro_threshold
            return z, empirical_p, is_normal
        return z, empirical_p

    def where_density_cache_nb_sims_lower_than(self, sizes, nb_sims):
        r'''
        Get the sizes whose density hasn't been simulated enough.

        Args:
            sizes (iterable): iterable of int values corresponding to sizes to test
            nb_sims (int): minimal number of simulations

        Returns:
            set:
                set of int values corresponding to sizes that haven't been simulated enough:

                .. math::
                    \{s \in \text{sizes} : |\text{density_cache}[s]| < \text{nb_sims}\}
        '''
        self.load_density_cache()
        return {size for size in sizes \
                     if size not in self.density_cache.keys() \
                     or len(self.density_cache[size]) < nb_sims}

    def where_lcc_cache_nb_sims_lower_than(self, sizes, nb_sims):
        r'''
        Get the sizes whose LCC hasn't been simulated enough.

        Args:
            sizes (iterable): iterable of int values corresponding to sizes to test
            nb_sims (int): minimal number of simulations

        Returns:
            set:
                set of int values corresponding to sizes that haven't been simulated enough:

                .. math::
                    \{s \in \text{sizes} : |\text{lcc_cache}[s]| < \text{nb_sims}\}
        '''
        self.load_lcc_cache()
        return {size for size in sizes \
                     if size not in self.lcc_cache.keys() \
                     or len(self.lcc_cache[size]) < nb_sims}

    def where_clustering_cache_nb_sims_lower_than(self, sizes, nb_sims):
        r'''
        Get the sizes whose clustering coefficient hasn't been simulated enough.

        Args:
            sizes (iterable): iterable of int values corresponding to sizes to test
            nb_sims (int): minimal number of simulations

        Returns:
            set:
                set of int values corresponding to sizes that haven't been simulated enough:

                .. math::
                    \{s \in \text{sizes} : |\text{clustering_cache}[s]| < \text{nb_sims}\}
        '''
        self.load_clustering_cache()
        return {size for size in sizes \
                     if size not in self.clustering_cache.keys() \
                     or len(self.clustering_cache[size]) < nb_sims}

    def fill_lcc_cache(self, nb_sims, sizes):
        r'''
        Fill the lcc_cache such that:

        .. math::
            \forall s \in \text{sizes} : |\text{lcc_cache[n]}| >= \text{nb_sims}

        Args:
            nb_sims (int): minimal number of simulations to be performed
            sizes (set): set of number of genes for which LCC size shall be tested
        '''
        self.load_lcc_cache()
        a = time()
        for idx, size in enumerate(sizes):
            self._compute_lcc_distribution(nb_sims, size)
            prop = (idx+1)/len(sizes)
            log('{} out of {}  ({:3.2f}%)    eta: {}' \
                .format(idx+1, len(sizes), 100*prop,
                        sec2date((time()-a)/prop*(1-prop))),
                end='\r')
        print('')
        self._write_lcc_cache()

    def fill_density_cache(self, nb_sims, sizes):
        r'''
        Fill the density cache such that:

        .. math::
            \forall s \in \text{sizes} : |\text{density_cache[n]}| \geq \text{nb_sims}

        Args:
            nb_sims (int): minimal number of simulations to be performed
            sizes (set): set of number of genes for which density shall be tested
        '''
        self.load_density_cache()
        a = time()
        for idx, size in enumerate(sizes):
            self._compute_disease_module_density(nb_sims, size)
            prop = (idx+1)/len(sizes)
            log('{} out of {}  ({:3.2f}%)    eta: {}' \
                .format(idx+1, len(sizes), 100*prop,
                        sec2date((time()-a)/prop*(1-prop))),
                end='\r')
        print('')
        self._write_density_cache()

    def fill_clustering_cache(self, nb_sims, sizes):
        r'''
        Fill the clustering cache such that:

        .. math::
            \forall s \in \text{ßizes} : |\text{clustering_cache[n]}| \geq \text{nb_sims}

        Args:
            nb_sims (int): minimal nuber of simulations to be performed
            sizes (set): set of number of genes for which clustering coefficient shall be tested
        '''
        self.load_clustering_cache()
        a = time()
        for idx, size in enumerate(sizes):
            self._compute_disease_modules_clustering(nb_sims, size)
            prop = (idx+1)/len(sizes)
            log('{} out of {}  ({:3.2f}%)    eta: {}' \
                .format(idx+1, len(sizes), 100*prop,
                        sec2date((time()-a)/prop*(1-prop))),
                end='\r')
        print('')
        self._write_clustering_cache()

    def get_subinteractome(self, genes, neighbourhood='none', namecode=None, neighb_count=1):
        r'''
        Extract a subinteractome and return it as an :class:`Interactome`
        object which is then usable for analyses.

        For :math:`H` a subgraph of :math:`G`, the first neighbourhood of :math:`H`
        within :math:`G` is defined by the graph:

        .. math::
            \mathcal N_G(H) = \Delta_{\mathcal N_G(V(H))}(G),

        where for every :math:`W \subset V(G)`:

        .. math::
            \mathcal N_G(W) = W \cup \left\{v \in V(G) : \exists w \in V(H) \text{ s.t. } \{v, w\} \in E(G)\right\} \subset V(G).

        Args:
            genes (set): the gene set inducing the subinteractome
            neighbourhood (str):
                one of the following: `'none'`, `'first'`, `'first-joined'` where:

                * `'none'` for no neighbouring gene
                * `'first'` for the first neighbourhood :math:`\mathcal N_G(H)` with :math:`G` being `self` and :math:`H` being `genes`
                * `'first-joined'` for the first neighbourhood with restriction that every neighbourhood gene must be associated to at least `neighb_count` genes.
            namecode (str): the namecode to be given to the subinteractome
            neighb_count (int): (only if `neighbourhood == 'first-joined'`) determines the minimum number of adjacent genes to be extracted:

                .. math::
                    \mathcal N_G^{(k)}(H) := \Delta_{\mathcal N_G^{(k)}}(H),

                with:

                .. math::
                    \mathcal N_G^{(k)}(W) := W \cup \left\{v \in V(G) : \exists \{v_1, \ldots, v_k\} \in \binom {V(H)}k \text{ s.t. } \{v, v_i\} \in E(G)
                        \quad (i=1, \ldots, k)\right\} \subset V(G).

        Return:
            :class:`Interactome`:
                the subinteractome
        '''
        #TODO: implement neighbourhood extraction
        genes &= self.genes
        genes_hash = md5(''.join(sorted(map(str, genes))).encode('utf-8')).hexdigest()
        path = self.interactome_path + genes_hash
        ret = IO.load_interactome(path, False, namecode)
        if ret is not None:
            return ret
        ret = deepcopy(self)
        ret.namecode = namecode
        ret.interactome_path = path

        ret.genes, ret.G = self._get_subinteractome_graph(genes, neighbourhood, neighb_count)
        print('So {} vertices, {} edges (density == {})' \
              .format(
                ret.G.num_vertices(),
                ret.G.num_edges(),
                2*ret.G.num_edges()/(ret.G.num_vertices()*(ret.G.num_vertices() - 1))
              )
        )
        genes_l = np.array(list(ret.genes))
        # Compute the mappings gene -> idx
        all_items = sorted(ret.genes2vertices.items(), key=lambda e: e[1])
        vp = ret.G.vp['genes']
        ret.genes2vertices = {vp[vertex]: int(vertex) for vertex in ret.G.vertices()}
        del ret.G.vertex_properties['genes']
        del self.G.vertex_properties['genes']
        ret.genes = set(ret.genes2vertices.keys())
        ret.lcc_cache = ret.density_cache = None
        ret.distances = None
        ret.compute_spls()
        IO.save_interactome(ret)
        return ret

    def _get_subinteractome_graph(self, genes, neighbourhood, neighb_count):
        print('Initially: {} genes'.format(len(genes)))
        if neighbourhood is not None and neighbourhood != 'none':
            genes = self._get_genes_neighbourhood(genes, neighbourhood, neighb_count)
        vp = self.G.new_vp('string')
        for gene in genes:
            vertex = self.genes2vertices[gene]
            vp[vertex] = gene
        self.G.vertex_properties['genes'] = vp
        genes_l = np.array(list(genes))
        # Extract subgraph with ``genes``
        G = self.get_subgraph(genes, True)
        # Ignore genes of degree 0
        genes_idx = np.where(G.get_out_degrees(np.arange(G.num_vertices())) > 0)[0]
        genes = {self.vertices2genes[idx] for idx in genes_idx}
        print('After removing isolated vertices: {} genes'.format(len(genes)))
        return genes, Graph(self.get_subgraph(genes, True), prune=True)

    def _get_genes_neighbourhood(self, genes, neighbourhood, neighb_count):
        raise NotImplementedError()
        # First neighbourhood
        vert2genes = dict()
        for k, v in self.genes2vertices.items():
            vert2genes[v] = k
        closure_genes = set()
        for gene in genes:
            gene_idx = self.genes2vertices[gene]
            for neighbour in self.G.get_out_neighbours(gene_idx):
                closure_genes.add(vert2genes[neighbour])
        return closure_genes | genes

    def copy(self):
        '''
        Return a copy of the interactome
        '''
        ret = deepcopy(self)
        ret.G = self.G.copy()  # watch out: deepcopy(self.G) returns None...
        return ret

    ##### Private methods

    def _compute_lcc_distribution(self, nb_sims, size):
        N = nb_sims
        if size in self.lcc_cache:
            nb_sims -= len(self.lcc_cache[size])
        if nb_sims < 0:
            print('[Warning]: {} sims required but {} already performed' \
                  .format(N, len(self.lcc_cache[size])))
            return
        lccs = np.empty(nb_sims, dtype=np.float)
        for i in range(nb_sims):
            lccs[i] = self.get_random_genes_lcc(size)
        if size in self.lcc_cache:
            self.lcc_cache[size] = np.concatenate((self.lcc_cache[size], lccs))
        else:
            self.lcc_cache[size] = lccs

    def _compute_disease_module_density(self, nb_sims, size):
        N = nb_sims
        if size in self.density_cache:
            nb_sims -= len(self.density_cache[size])
        if size <= 0 or nb_sims <= 0:
            return
        densities = np.empty(nb_sims, dtype=np.float)
        for i in range(nb_sims):
            densities[i] = self.get_random_genes_density(size)
        try:
            densities = np.concatenate((self.density_cache[size], densities))
        except (KeyError, ValueError):
            pass
        self.density_cache[size] = densities

    def _compute_disease_modules_clustering(self, nb_sims, size):
        N = nb_sims
        if size in self.clustering_cache:
            nb_sims -= len(self.clustering_cache[size])
        if size < 3 or nb_sims <= 0:
            return
        clustering_coeffs = np.empty(nb_sims, dtype=np.float)
        for i in range(nb_sims):
            clustering_coeffs[i] = self.get_random_genes_clustering(size)
        try:
            clustering_coeffs = np.concatenate((self.clustering_cache[size], clustering_coeffs))
        except (KeyError, ValueError):
            pass
        self.clustering_cache[size] = clustering_coeffs

    def _write_lcc_cache(self):
        IO.save_lcc_cache(self, self.lcc_cache)

    def _write_density_cache(self):
        IO.save_density_cache(self, self.density_cache)

    def _write_clustering_cache(self):
        IO.save_clustering_cache(self, self.clustering_cache)

    def save(self):
        IO.save_interactome(self)
