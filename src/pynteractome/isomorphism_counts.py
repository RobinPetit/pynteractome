from pynteractome.core.analyses.isomorphism import are_isomorphic

class IsomorphismCounts:
    r'''
    Representation of :math:`\Omega_\alpha` (see
    :func:`pathogenic_topology_analysis <pynteractome.core.analyses.topology.pathogenic_topology_analysis>`
    for definition).
    '''
    def __init__(self):
        self.keys = list()
        self.values = list()

    def __iter__(self):
        return self.keys.__iter__()

    def __len__(self):
        return self.keys.__len__()

    def items(self):
        return zip(self.keys, self.values)

    def add(self, graph, term, genes):
        r'''
        Add a 3-uple :math:`(\Xi_\alpha(\sigma), t, \sigma)` where ``genes`` is :math:`\sigma`, ``graph``
        is :math:`\Xi_\alpha(\sigma)` and ``term`` is :math:`t`.

        Note: this tuple being in :math:`\Omega_\alpha` (so an ``IsomorphismCounts`` object indicates that
        :math:`(t, \sigma) \in \Lambda_\alpha`, i.e. :math:`\sigma \in \mathcal P_\alpha(\mu(t))`.
        '''
        idx = self._get_idx(graph)
        if idx == -1:
            self._append(graph, {term: [genes]})
        else:
            if term in self.values[idx]:
                self.values[idx][term].append(genes)
            else:
                self.values[idx][term] = [genes]

    def contains(self, graph):
        r'''
        Return ``True`` if ``graph`` (:math:`= \Xi_\alpha(\sigma)`) is contained within the object
        and False ``otherwise``.
        '''
        return self._get_idx(graph) == -1

    def _get_idx(self, graph):
        for i, G in enumerate(self.keys):
            if are_isomorphic(G, graph):
                return i
        return -1

    def _append(self, graph, d):
        '''
        Args:
            d (dict):
                Mapping HPO terms -> list of gene subsets
        '''
        self.keys.append(graph)
        self.values.append(d)

    def merge(self, iso_counts):
        '''
        Args:
            iso_counts (:class:`IsomorphismCounts <pynteractome.isomorphism_counts.IsomorphismCounts>`):
                Object to merge with ``self``.
        '''
        for i, G in enumerate(iso_counts):
            idx = self._get_idx(G)
            if idx == -1:
                self._append(G, iso_counts.values[i])
            else:
                for (t, genes_subsets) in iso_counts.values[i].items():
                    if t in self.values[idx]:
                        self.values[idx][t].extend(genes_subsets)
                    else:
                        self.values[idx][t] = genes_subsets

