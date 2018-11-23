class IntegratorConfig:
    def __init__(self, interactome_path, disease_genes_path,
                 disease_names_path, gene2omim_path):
        self.interactome_path = interactome_path
        self.disease_genes_path = disease_genes_path
        self.disease_names_path = disease_names_path
        self.gene2omim_path = gene2omim_path
