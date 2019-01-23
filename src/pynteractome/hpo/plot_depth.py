import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pynteractome.hpo.load_obo import load_abnormal_phenotypes

def plot_depth(hpo):
    sp = nx.shortest_path_length(hpo, source=118)
    v = [0]*(max(sp.values())+1)
    for term in sp:
        v[sp[term]] += 1
    V = list()
    for value, count in enumerate(v):
        print('(depth, count) == {}'.format((value, count)))
        V.extend([value] * count)
    print('median: {}\nmean: {}'.format(np.median(V), np.mean(V)))
    v = np.asarray(v, np.float)
    v /= v.sum()
    plt.bar(range(len(v)), v, align='center', color='salmon')
    plt.yscale('log')
    plt.xticks(np.arange(len(v)))
    plt.xlim(-1, len(v))
    plt.xlabel('Term depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of HPO terms depth')
    plt.show()

def plot_hpo_depth():
    hpo = load_abnormal_phenotypes()
    plot_depth(hpo)

if __name__ == '__main__':
    plot_hpo_depth()
