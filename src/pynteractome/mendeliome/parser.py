from pynteractome.utils import reverse_set_dict

class MendeliomeParser():
    '''
    MendeliomeParser is an interface class for the database
    file and the application.
    '''
    def __init__(self, diseases_genes_path, disease_names_path, gene2omim_path):
        '''
        diseases_path: the path (relative or absolute) to the db diseases file
        '''
        self.disease_abbr = MendeliomeParser.load_disease_abbr(disease_names_path)
        self.diseases_genes_path = diseases_genes_path
        self.disease_genes = self.load_file(diseases_genes_path)
        self.disease_genes_dict = dict()
        for gene, disease in self.disease_genes:
            disease = self.disease_abbr[disease]
            if disease in self.disease_genes_dict:
                self.disease_genes_dict[disease].add(gene)
            else:
                self.disease_genes_dict[disease] = {gene}
        self.diseases = list(self.disease_genes_dict.keys())  # disease panels
        # load a dict matching a gene to the set of associated phenotypes
        self.gene2omim = MendeliomeParser.load_gene2omim(gene2omim_path)
        self.omim2gene = reverse_set_dict(self.gene2omim)

    @staticmethod
    def load_gene2omim(path):
        gene2omim = dict()
        couples = MendeliomeParser.load_file(path)
        couples = [couple for couple in couples if len(couple) == 2]
        for gene, omim in couples:
            omim = MendeliomeParser.get_omim(omim)
            if not isinstance(omim, int):
                continue
            gene = int(gene)
            if gene not in gene2omim:
                gene2omim[gene] = {omim}
            else:
                gene2omim[gene].add(omim)
        return gene2omim

    @staticmethod
    def get_omim(omim):
        PREFIX = 'OMIM:'
        try:
            return int(omim)
        except ValueError:
            if not isinstance(omim, str):
                return None
            if omim.startswith(PREFIX):
                try:
                    return int(omim[len(PREFIX):])
                except ValueError:
                    return None

    @staticmethod
    def load_disease_abbr(path):
        couples = MendeliomeParser.load_file(path)
        ret = dict()
        for el in couples:
            if len(el) == 1:
                ret[el[0]] = el[0]
            else:
                ret[el[0]] = el[1]
        return ret

    @staticmethod
    def load_file(path):
        content = list()
        with open(path) as diseases_file:
            for line in diseases_file:
                if line[0] == '#':
                    continue
                content.append(line.strip().upper().split('\t'))
        return content

    def get_omim2gene(self):
        return self.omim2gene

    def get_gene2omim(self):
        return self.gene2omim

    def get_all_diseases(self):
        '''
        return all the diseases that are in Mendeliome
        '''
        return set(self.diseases)

    def get_disease_genes(self, disease):
        '''
        return all the associated genes with the precised disease

        disease: the name of the disease
        '''
        assert disease in self.get_all_diseases()
        return self.disease_genes_dict[disease]

    def get_phenotypes_of_gene(self, gene):
        return self.gene2omim[gene]

    def get_all_genes(self):
        ret = set()
        for disease in self.get_all_diseases():
            ret = ret | self.get_disease_genes(disease)
        return set(map(int, ret))

    def get_all_disease_genes(self):
        return self.disease_genes

    def get_disease_genes_dict(self):
        '''
        return a dictionary with str keys containing the disease names and
        with set values containing the associated genes.

        Made up example:
            >>> mp = MendeliomeParser(path)
            >>> disease_genes_associations = mp.get_disease_genes_dict()
            >>> for disease, genes in disease_genes_associations.items():
            ...     print('disease:', disease)
            ...     print('associated genes:', genes)
            ...     print('-'*30)
            disease: CANCER
            associated genes: {1234, 2345, 3456}
            ------------------------------
            disease: AUTISM
            associated genes: {2056, 3141, 2718, 6283, 9876}
            ------------------------------
        '''
        return self.disease_genes_dict
