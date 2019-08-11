from pynteractome.hpo.plot_height import get_heights
from pynteractome.hpo.plot_depth import get_depths
from pynteractome.hpo.load_obo import load_abnormal_phenotypes

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    hpo = load_abnormal_phenotypes()
    heights = get_heights(hpo)
    depths = get_depths(hpo)
    assert set(depths.keys()) == set(heights.keys())
    depths, heights = map(np.asarray, zip(*list({k: (depths[k], heights[k]) for k in depths}.values())))
    print(stats.spearmanr(depths, heights))
    print(np.corrcoef(depths, heights)[0,1])
    table = np.zeros((np.unique(depths).size, np.unique(heights).size), dtype=np.int)
    for d, h in zip(depths, heights):
        table[d,h] += 1
    print(table)
    print(stats.chi2_contingency(table)[:3])
