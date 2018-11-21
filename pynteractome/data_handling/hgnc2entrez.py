HGNC_PATH = '../../data/hgnc2entrez.tsv'

hgnc2entrez = None

def load_hgnc2entrez():
    global hgnc2entrez
    hgnc2entrez = dict()
    with open(HGNC_PATH) as hgnc_file:
        hgnc_file.readline()  # ignore first line
        for l in hgnc_file:
            cols = l[:-1].split('\t')  # remove '\n' at the EOL
            if len(cols) != 5:
                continue
            if cols[4] == '':
                continue
            hgnc2entrez[cols[0].upper()] = cols[4]
            for previous in cols[2].split(', '):
                hgnc2entrez[previous.upper()] = cols[4]
            for synonym in cols[3].split(', '):
                hgnc2entrez[synonym.upper()] = cols[4]

def get_hgnc2entrez():
    if hgnc2entrez is None:
        load_hgnc2entrez()
    return hgnc2entrez
