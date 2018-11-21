#!/usr/bin/python3

from .hgnc2entrez import get_hgnc2entrez

DATA_DIR = '../../data/'

hgnc2entrez = get_hgnc2entrez()

file_header = list()
new_file_content = list()
unhandled_genes = set()
with open(DATA_DIR + 'disease genes.tsv') as diseage_genes_file:
    for line in diseage_genes_file:
        if line.startswith('#'):
            file_header.append(line)
            continue
        cols = line[:-1].split('\t')
        cols[0] = cols[0].upper()
        if cols[0] not in hgnc2entrez:
            unhandled_genes.add(cols[0])
        else:
            new_file_content.append(hgnc2entrez[cols[0]] + '\t' + cols[1] + '\n')
with open(DATA_DIR + 'converted disease genes.tsv', 'w') as new_disease_genes_file:
    new_disease_genes_file.write(''.join(file_header))
    new_disease_genes_file.writelines(new_file_content)
with open(DATA_DIR + 'unhandled genes.txt', 'w') as unhandled_genes_file:
    unhandled_genes_file.write('Not recognized genes:\n')
    unhandled_genes_file.write('\n'.join(sorted(unhandled_genes)))
