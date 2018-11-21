with open('../data/hpo/phenotype_annotation.tsv') as hpo_file:
    content = [l.upper().split('\t') for l in hpo_file]
omim_lines = [l for l in content if 'OMIM:' in l[5] and l[5] != 'OMIM:']
for omim_line in range(omim_lines):
    sources = omim_line[5].replace(';', ',').split(',')
    for src in sources:
        if src.startswith('OMIM:'):
            omim_line[5] = src
            break

print('{} elements related to OMIM'.format(len(omim_lines)))
with open('../data/hpo/omim_phenotype_annotation.tsv', 'w') as output_file:
    output_file.write(''.join(['\t'.join(l) for l in omim_lines]))

col5 = set([l[5][len('OMIM:'):] for l in omim_lines])
print(len(col5))
with open('omim-related-hpo.txt', 'w') as output_file:
    output_file.write('\n'.join(sorted(col5)))
