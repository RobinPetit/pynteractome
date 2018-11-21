def get_omim2hpo(path='../data/hpo/omim_phenotype_annotation.tsv'):
    omim_to_hpo = dict()
    with open(path) as annotations:
        for line in annotations:
            cols = line.split('\t')
            omim = int(cols[5][len('OMIM:'):])
            hpo = int(cols[4][len('HP:'):])
            if omim in omim_to_hpo:
                omim_to_hpo[omim].add(hpo)
            else:
                omim_to_hpo[omim] = {hpo}
    return omim_to_hpo
