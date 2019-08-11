import numpy as np
import matplotlib.pyplot as plt
from pynteractome.hpo.load_obo import load_abnormal_phenotypes

INF = float('+inf')

def annotate(G, term, heights):
    if term in heights:
        return heights[term]
    min_child_height = INF
    for child in G[term]:
        child_height = annotate(G, child, heights)
        if child_height < min_child_height:
            min_child_height = child_height
    heights[term] = 0 if min_child_height == INF else min_child_height+1
    return heights[term]

def get_heights(hpo):
    heights = dict()
    annotate(hpo, 118, heights)
    return heights

def plot_height(heights):
    print('mean: {}\nmedian: {}'.format(heights.mean(), np.median(heights)))
    values, counts = np.unique(heights, return_counts=True)
    ys = np.zeros(values.max()+1, dtype=np.double)
    for x, y in zip(values, counts):
        ys[x] = y
    ys /= ys.sum()
    plt.bar(np.arange(ys.size), ys, align='center', color='salmon')
    plt.xticks(np.arange(ys.size))
    plt.xlabel('Term height')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.xlim(-1, ys.size)
    plt.title('Distribution of HPO terms height')
    plt.show()

if __name__ == '__main__':
    hpo = load_abnormal_phenotypes()
    heights = get_heights(hpo)
    plot_height(np.asarray(list(heights.values())))
