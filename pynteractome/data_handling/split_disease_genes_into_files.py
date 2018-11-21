#!/usr/bin/python3

from sys import argv

CHARS_TO_REMOVE = '()+/?'
DATA_DIR = '../../data/'

def split_classes():
    disease2genes = dict()
    with open(DATA_DIR + 'converted disease genes.tsv') as disease_genes_file:
        for line in disease_genes_file:
            if line.startswith('#'):
                continue
            gene, disease = line.strip().split('\t')
            if disease in disease2genes:
                disease2genes[disease].append(gene)
            else:
                disease2genes[disease] = [gene]
    for disease in disease2genes:
        disease_path = ''.join(c for c in disease.replace(' ', '_') if c not in CHARS_TO_REMOVE)
        with open(DATA_DIR + 'classes/' + disease_path + '.txt', 'w') as disease_file:
            disease_file.write('\n'.join(disease2genes[disease]))

def split_phenotypes():
    THRESHOLD = 10
    disease2genes = dict()
    with open(DATA_DIR + 'converted gene2omim.tsv') as disease_genes_file:
        for line in disease_genes_file:
            if line.startswith('#'):
                continue
            gene, disease = line.strip().split('\t')
            if disease in disease2genes:
                disease2genes[disease].append(gene)
            else:
                disease2genes[disease] = [gene]
    for disease in disease2genes:
        if len(disease2genes[disease]) < THRESHOLD:
            continue
        disease_path = ''.join(c for c in disease.replace(' ', '_') if c not in CHARS_TO_REMOVE)
        with open(DATA_DIR + 'phenotypes/' + disease_path + '.txt', 'w') as disease_file:
            disease_file.write('\n'.join(disease2genes[disease]))

if __name__ == '__main__':
    if len(argv) < 2:
        print('usage: ./split_disease_genes_into_files.py [classes or phenotypes]')
        exit()
    {
        'classes': split_classes,
        'phenotypes': split_phenotypes
    }[argv[1]]()
