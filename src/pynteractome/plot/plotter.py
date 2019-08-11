import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# local
from pynteractome.IO import IO
from pynteractome.utils import extract_triangular, fmt_g, fmt_e, log
from pynteractome.warning import warning

#plt.rc('text', usetex=True)  # Activate LaTeX rendering

DEFAULT_PLOT_DIR = '../plots/'

AVAILABLE_FORMATS = (
    'eps',
    'png',
)

class Plotter:
    _PLOT_DIR = None

    @staticmethod
    def _set_plot_dir(integrator):
        plot_dir = DEFAULT_PLOT_DIR
        namecode = integrator.interactome.namecode
        if namecode is None:
            warning('Plotting with unknown interactome. ' + \
                    'Default plot dir is used: + "' + DEFAULT_PLOT_DIR + '"')
        else:
            plot_dir += namecode + '/'
        Plotter._PLOT_DIR = plot_dir

    @staticmethod
    def _get_plot_dir_or_die():
        if Plotter._PLOT_DIR is None:
            raise ValueError(
                '[Plotter] Plots dir has not been set yet. ' + \
                'You are probably using :class:`Plotter` the wrong way. ' + \
                'Only call existing methods from this')
        return Plotter._PLOT_DIR

    @staticmethod
    def save_fig(fig, path):
        plot_dir = Plotter._get_plot_dir_or_die()
        path = plot_dir + path
        ridx = path.rfind('.')
        if ridx > 0:
            ext = path[ridx+1:]
            if ext not in AVAILABLE_FORMATS:
                warning('Unknown format: "{}". Setting default format ("{}").' \
                        .format(ext, AVAILABLE_FORMATS[0]))
                path = path[:ridx+1] + AVAILABLE_FORMATS[0]
        log('Saving figure to path "{}"'.format(path))
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_clustering(integrator, gene_mapping):
        Plotter._set_plot_dir(integrator)
        cache = integrator.interactome.get_clustering_cache()
        ps = list()
        hpo2genes = integrator.get_hpo2genes(gene_mapping)
        for term, genes in hpo2genes.items():
            genes &= integrator.interactome.genes
            N = len(genes)
            if N < 3:
                continue
            c = integrator.interactome.get_genes_clustering(genes, entrez=True)
            if np.isnan(c):
                print('C is ill defined on HPO term {}'.format(term))
                continue
            k = np.isnan(cache[N]).sum()
            cache[N][np.isnan(cache[N])] = 0
            if np.isnan(cache[N]).any():
                print('Still NaN')
            p = (cache[N] >= c).sum() / len(cache[N])
            ps.append(p)
        print('{} ps are still available'.format(len(ps)))
        print(np.unique(ps))
        ps = np.asarray(ps)
        logps = np.log10(ps)
        logps[ps == 0] = -10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([np.log10(.05)]*2, [0, 100], 'k-.', label=r'$p = .05$')
        Plotter.plot_pdf_and_cdf(logps, 20, 'salmon', 'r', 'log10(p)', ax=ax, remove_ticks=False)
        Plotter.save_fig(fig, 'clustering.eps')

    @staticmethod
    def loc_hpo(integrator, gene_mapping):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        integrator.reset_propagation()
        for depth in [0, integrator.get_hpo_depth()]:#range(integrator.get_hpo_depth()):
            integrator.propagate_genes(depth)
            hpo2genes = integrator.get_hpo2genes(gene_mapping)
            zs = dict()
            #for (term, genes) in integrator.get_hpo2genes(gene_mapping).items():
            for term in integrator.iter_leaves(1):
                if term not in hpo2genes:
                    continue
                genes = hpo2genes[term]
                if len(genes) > 1:
                    zs[term] = interactome.get_lcc_score(genes, 0, shapiro=True)
            Plotter._loc_hpo(integrator, zs, depth, gene_mapping)

    @staticmethod
    def _loc_hpo(integrator, zs, prop_depth, gene_mapping):
        Plotter._set_plot_dir(integrator)
        integrator.propagate_genes(prop_depth)
        interactome = integrator.interactome
        hpo2genes = integrator.get_hpo2genes(gene_mapping)
        xs, ys = list(), list()
        are_normal = list()
        empirical_ps = list()
        for term, (z_score, empirical_p, is_normal) in zs.items():
            if term not in hpo2genes:
                continue
            if z_score is not None:
                z = float(z_score)
                genes = hpo2genes[term] & interactome.genes
                if len(genes) > 1:
                    lcc = interactome.get_genes_lcc_size(interactome.verts_id(genes))
                    rel_size = lcc / len(genes)
                    xs.append(rel_size)
                    ys.append(z)
                    are_normal.append(is_normal)
                    empirical_ps.append(empirical_p)
        print('')
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        are_normal = np.asarray(are_normal)
        empirical_ps = np.asarray(empirical_ps)
        Plotter.significance_bar_plot(
            ys, empirical_ps,
            'Significance via $p_{emp}$ or $z$ (HPO terms)',
            'loc/barplot.significance.hpo.{}.{}.eps'.format(prop_depth, gene_mapping)
        )
        print(len(set(np.where(np.logical_and(ys >= 1.65, empirical_ps >= .05))[0])),
              'terms have significant z but non-significant p')
        print(len(set(np.where(np.logical_and(ys < 1.65, empirical_ps < .05))[0])),
              'terms have non-significant z but significant p')
        if prop_depth == 0:
            title = 'Significance of |LCC| (non-propagated - {})'.format('$\gamma_\cup$' if gene_mapping == 'union' else '$\gamma_\cap$')
        elif prop_depth == integrator.get_hpo_depth():
            title = 'Significance of |LCC| (fully up-propagated - {})'.format('$\gamma_\cup$' if gene_mapping == 'union' else '$\gamma_\cap$')
        else:
            title = 'Significance of |LCC| (up-propagated by {} - {})'.format(prop_depth, '$\gamma_\cup$' if gene_mapping == 'union' else '$\gamma_\cap$')
        empirical_ps[empirical_ps < 1e-10] = 1e-10
        empirical_ps = np.log10(empirical_ps)
        path = 'loc/prop.depth.{}.z.{}.eps'.format(prop_depth, gene_mapping)
        Plotter._plot_loc_zs(xs, ys, title, path, are_normal)
        path = 'loc/prop.depth.{}.empirical.p.{}.eps'.format(prop_depth, gene_mapping)
        Plotter.plot_z_vs_empirical_p(ys, empirical_ps, title, path)
        path = 'loc/prop.depth.{}.p.{}.eps'.format(prop_depth, gene_mapping)
        Plotter._plot_loc_ps(xs, empirical_ps, title, path, are_normal)

    @staticmethod
    def loc_omim(integrator):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        omim2genes = integrator.get_omim2genes()
        xs, ys = list(), list()
        are_normal = list()
        empirical_ps = list()
        for genes in omim2genes.values():
            genes &= interactome.genes
            if not genes or len(genes) <= 1:
                continue
            z, empirical_p, shapiro_p = interactome.get_lcc_score(genes, 0, shapiro=True)
            if z is None:
                continue
            lcc = interactome.get_genes_lcc_size(interactome.verts_id(genes))
            rel_size = lcc / len(genes)
            assert rel_size <= 1
            xs.append(rel_size)
            ys.append(z)
            are_normal.append(shapiro_p >= .05)
            empirical_ps.append(empirical_p)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        are_normal = np.asarray(are_normal)
        empirical_ps = np.asarray(empirical_ps)
        Plotter.significance_bar_plot(
            ys, empirical_ps,
            r'Significance via $p$ or $z$ (OMIM diseases)',
            'barplot.significance.omim.eps'
        )
        empirical_ps[empirical_ps < 1e-10] = 1e-10
        empirical_ps = np.log10(empirical_ps)
        title = 'Significance of |LCC| (OMIM diseases)'
        path = 'loc/omim.z.eps'
        Plotter._plot_loc_zs(xs, ys, title, path, are_normal)
        path = 'loc/omim.empirical.p.eps'
        Plotter.plot_z_vs_empirical_p(ys, empirical_ps, title, path)
        path = 'loc/omim.p.eps'
        Plotter._plot_loc_ps(xs, empirical_ps, title, path, are_normal)

    @staticmethod
    def _plot_loc_zs(xs, zs, title, path, are_normal):
        print('{}/{} are significant'.format((zs > 1.65).sum(), len(zs)))
        print('{}/{} are < 0'.format((zs < 0).sum(), len(zs)))
        print('{} out of {} are normal'.format(are_normal.sum(), len(are_normal)))
        fig, axes = Plotter.dot_plot_with_hists(
            xs, zs, 'Relative size: |LCC|/|S|', 'z-score', title, figsize=(6, 6)
        )
        ax = axes[0]
        ax.plot([-0., 1], [1.65]*2, 'k-.')
        ax.plot([0, 1], [-1.65]*2, 'k-.')
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_xticks(np.arange(0, 11, 2)/10)
        ax.set_xticklabels(map(fmt_g, ax.get_xticks()))
        ax.get_xticklabels()[-1].set_ha('right')
        ax.set_yticklabels(map(fmt_g, ax.get_yticks()))
        axes[2].set_yticks(ax.get_yticks())
        axes[2].set_ylim(ax.get_ylim())
        axes[1].set_xticks(ax.get_xticks())
        axes[1].set_xlim(ax.get_xlim())
        Plotter.save_fig(fig, path)

    @staticmethod
    def plot_z_vs_empirical_p(zs, ps, title, path):
        fig, axes = Plotter.dot_plot_with_hists(zs, ps, 'z-score', 'log10(Empirical p)', title, figsize=(6, 6))
        ax = axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(xlim, [np.log10(.05)]*2, 'k-.')
        ax.plot([1.65, 1.65], ylim, 'k-.')
        ax.grid(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        Plotter.save_fig(fig, path)

    @staticmethod
    def _plot_loc_ps(xs, empirical_ps, title, path, are_normal):
        print('According to Shapiro-Wilk: {}/{} are normal'.format(are_normal.sum(), len(are_normal)))
        print('{}/{} are significant'.format((empirical_ps < .05).sum(), len(empirical_ps)))
        fig, axes = Plotter.dot_plot_with_hists(
            xs, empirical_ps, 'Relative size: |LCC|/|S|', 'log10(Empirical p)', title, figsize=(6, 6)
        )
        ax = axes[0]
        xlim = [0, 1]
        ax.plot(xlim, [np.log10(.05)]*2, 'k-.')
        ax.grid(True)
        ax.set_xlim(xlim)
        axes[1].set_xticks(ax.get_xticks())
        axes[1].set_xlim(ax.get_xlim())
        Plotter.save_fig(fig, path)

    @staticmethod
    def significance_bar_plot(zs, empirical_ps, title, path):
        non_significant = np.where(np.logical_and(zs <= 1.65, empirical_ps >= .05))[0]
        z_significant = np.where(np.logical_and(zs > 1.65, empirical_ps >= .05))[0]
        p_significant = np.where(np.logical_and(zs <= 1.65, empirical_ps < .05))[0]
        significant = np.where(np.logical_and(zs > 1.65, empirical_ps < .05))[0]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        lengths = np.asarray(
            [len(non_significant), len(z_significant), len(p_significant), len(significant)]
        )
        lengths = 100*lengths / lengths.sum()
        ax.bar(
            np.arange(len(lengths)), lengths, align='center', color='salmon',
            tick_label=(
                '$z \leq 1.65$\n$p_{emp} \geq 0.05$',
                '$z > 1.65$\n$p_{emp} \geq 0.05$',
                '$z \leq 1.65$\n$p_{emp} < 0.05$',
                '$z > 1.65$\n$p_{emp} < 0.05$'
            )
        )
        for x, y in zip(ax.get_xticks(), lengths):
            ax.text(x-.1, y+5, '{:.2f}%'.format(y), weight='bold')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Frequency')
        #ax.grid(True)
        Plotter.save_fig(fig, path)

    @staticmethod
    def sep(integrator):
        Plotter._set_plot_dir(integrator)
        integrator.reset_propagation()
        for prop_depth in [0, integrator.get_hpo_depth()]:
            integrator.propagate_genes(prop_depth)
            nb_genes = integrator.get_nb_associated_genes()
            matrix_seps, matrix_Cs = IO.load_sep(integrator.interactome, 0, prop_depth)[:2]
            if matrix_seps.shape == (0, 0) or matrix_Cs.shape == (0, 0):
                continue
            prop = 'prop' if prop_depth > 0 else 'no-prop'
            for min_nb_genes in [0, 10]:
                indices = nb_genes >= min_nb_genes
                seps, Cs = map(lambda a: extract_triangular(a, indices), [matrix_seps, matrix_Cs])
                print('{}|{}\t\t[Pearson] (r, p): {}\t\t[Spearman] (r, p): {}' \
                      .format(prop_depth, min_nb_genes,
                              stats.pearsonr(seps, Cs), stats.spearmanr(seps, Cs)))
                Plotter._plot_sep_hist(seps[Cs == 0], 'no overlap', prop, min_nb_genes)
                Plotter._plot_sep_hist(
                    seps[np.logical_and(Cs != 0, Cs != 1)],
                    'partial overlap', prop, min_nb_genes)
                Plotter._plot_sep_hist(seps[Cs == 1], 'complete overlap', prop, min_nb_genes)
                Plotter._plot_sep_hist(seps, 'summary', prop, min_nb_genes)
                Plotter.KS_sep_families(seps, Cs)

    @staticmethod
    def KS_sep_families(seps, Cs):
        xs = [seps[Cs == 0], seps[np.logical_and(Cs != 0, Cs != 1)], seps[Cs == 1], seps]
        for i in range(len(xs)):
            xs[i] = (xs[i] - xs[i].mean()) / xs[i].std()
        output = np.empty((len(xs), len(xs)), dtype=np.float)
        for i in range(len(xs)):
            output[i,i] = 1
            for j in range(i+1, len(xs)):
                output[i,j] = output[j,i] = stats.ks_2samp(xs[i], xs[j])[1]
        print(output)

    @staticmethod
    def _plot_sep_hist(array, name, prop, min_nb_genes):
        colors = {
            'no overlap': 'tomato',
            'partial overlap': 'paleturquoise',
            'complete overlap': 'limegreen',
            'summary': 'orchid'
        }
        print('\t\t', name, prop, min_nb_genes)
        print(
            ('# < 0: {:,d} ({:3.2f}%)\n' + \
             '# = 0: {:,d} ({:3.2f}%)\n' + \
             '# > 0: {:,d} ({:3.2f}%)\n' + \
             '\ttotal: {:,d}') \
            .format(
                (array < 0).sum(), 100 * (array < 0).sum() / array.size,
                (array == 0).sum(), 100 * (array == 0).sum() / array.size,
                (array > 0).sum(), 100 * (array > 0).sum() / array.size,
                array.size
            )
        )
        NB_BINS = 50
        bins = np.arange(NB_BINS+1)/(NB_BINS+1)*9 - 5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(array, bins, color=colors[name])
        ax.set_title(
            name + '\n{:,d} < 0 and {:,d} > 0'.format((array < 0).sum(), (array > 0).sum())
        )
        y_max = 1e7
        ax.set_ylim((.9, y_max))
        ax.plot([0, 0], [.9, y_max], 'k-.')
        ax.set_yscale('log')
        path = 'sep-{}-{}-{}.eps'.format(min_nb_genes, prop, name.replace(' ', '-'))
        Plotter.save_fig(fig, path)

    @staticmethod
    def gamma_density(integrator, gene_mapping):
        Plotter._set_plot_dir(integrator)
        non_empty_links = list()
        term2genes = integrator.get_hpo2genes(gene_mapping)
        for term in term2genes:
            genes = term2genes[term] & integrator.interactome.genes
            if len(genes) > 20:
                non_empty_links.append(len(genes))
        print(np.max(non_empty_links))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(non_empty_links, 200)
        Plotter.save_fig(fig, 'gamma_density.eps')

    @staticmethod
    def relation_degree_nb_terms(integrator, depth, gene_mapping='intersection'):
        Plotter._set_plot_dir(integrator)
        integrator.propagate_genes(depth)
        degrees = list()
        terms_prop = list()
        gene2hpo = integrator.get_gene2hpo(gene_mapping)
        for gene in gene2hpo:
            try:
                vert_id = integrator.interactome.vert_id(gene)
            except:
                continue
            nb_terms_propagated = len(gene2hpo[gene])
            degree = integrator.interactome.G.get_out_degrees([vert_id])
            degrees.append(degree)
            terms_prop.append(nb_terms_propagated)
        for log in (True, False):
            Plotter._plot_deg_vs_nb_terms(degrees, terms_prop, color='b', path=str(depth), log=log)

    @staticmethod
    def _plot_deg_vs_nb_terms(degrees, terms, color, path, log=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(degrees, terms, color+'o')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Number of related terms')
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            path = 'log-' + path
        if not path.endswith('.eps'):
            path += '.eps'
        Plotter.save_fig(fig, path)

    @staticmethod
    def plot_deg_distribution(integrator):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        degs = interactome.G.get_out_degrees(np.arange(interactome.G.num_vertices()))
        degrees = np.arange(degs.max())+1
        print('skewness:', stats.skew(degs))
        print('kurtosis:', stats.kurtosis(degs))
        print('max degree:', degrees[-1])
        print(stats.kurtosistest(degs))
        print(stats.kstest(degs, 'lognorm', N=1000, args=(degs.mean(), degs.std())))
        counts = np.zeros(degrees.shape, dtype=np.float)
        for d in degs:
            if d == 0:
                continue
            counts[int(d)-1] += 1
        counts /= counts.sum()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(degrees[counts != 0], counts[counts != 0], 'ro', mec='red', ms=3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree distribution of the {}'.format({'default': 'interactome', 'mendeliome': 'mendeliome'}[interactome.namecode]))
        Plotter.save_fig(fig, 'deg_distribution.eps')

    @staticmethod
    def hpo_to_omim(integrator):
        Plotter._set_plot_dir(integrator)
        hpo2omim = integrator.get_hpo2omim()
        depth = integrator.get_hpo_depth()
        print(depth)
        fig = plt.figure()
        counter = 1
        for n in range(depth):
            nb_associations = [len(hpo2omim[term]) for term in integrator.order_n_ontology(n) \
                                                   if term in hpo2omim]
            if len(nb_associations) == 0:
                continue
            ax = fig.add_subplot(4, 4, counter)
            counter += 1
            ax.hist(nb_associations, bins=25, color='salmon',
                    weights=np.ones(len(nb_associations)) / len(nb_associations),
                    label='depth: {}'.format(n))
            ax.set_yscale('log')
            if counter % 3 == 2:
                ax.set_ylabel('Frequency')
            if counter // 3 == 3:
                ax.set_xlabel('Number of OMIM phenotypes')
            ax.legend()
        Plotter.save_fig(fig, 'hpo_to_omim.eps')

    @staticmethod
    def hpo_to_omim_mean_median(integrator):
        Plotter._set_plot_dir(integrator)
        hpo2omim = integrator.get_hpo2omim()
        fig = plt.figure(figsize=(10, 6))
        depth = integrator.get_hpo_depth()
        print(depth)
        ax = fig.add_subplot(111)
        all_values = [
            [len(hpo2omim[term]) for term in integrator.order_n_ontology(n) if term in hpo2omim] \
                for n in range(depth)
        ]
        indices = list()
        vs = list()
        for i, v in enumerate(all_values):
            if len(v) != 0:
                indices.append(i)
                vs.append(v)
        ax.plot(
            indices, [np.median(v) for v in vs], 'r:',
            marker='x', ms=15, label='median (original)'
        )
        ax.plot(
            indices, [np.mean(v) for v in vs], 'b:',
            marker='x', ms=15, label='mean (original)'
        )
        integrator.propagate_genes(integrator.get_hpo_depth())
        hpo2omim = integrator.get_hpo2omim()
        all_values = [
            [len(hpo2omim[term]) for term in integrator.order_n_ontology(n) if term in hpo2omim] \
                for n in range(integrator.get_hpo_depth())
        ]
        indices = list()
        vs = list()
        for i, v in enumerate(all_values):
            if len(v) != 0:
                indices.append(i)
                vs.append(v)
        ax.plot(
            indices, [np.median(v) for v in vs], 'r:',
            marker='*', ms=15, label='median (propagated)'
        )
        ax.plot(
            indices, [np.mean(v) for v in vs], 'b:',
            marker='*', ms=15, label='mean (propagated)'
        )
        ax.set_yscale('log')
        ax.set_xlabel('Depth')
        ax.set_title('Distribution of the number of OMIM associations per term')
        ax.set_ylabel('Number of OMIM associations')
        ax.set_xlim(min(indices)-1, max(indices)+1)
        ax.set_xticks(indices)
        ax.legend(loc='upper right')
        ax.grid(True)
        Plotter.save_fig(fig, 'hpo_to_omim_mean_median.eps')

    @staticmethod
    def dot_plot_with_hists(xs, ys, xlabel, ylabel, title, grid=False, figsize=(12.5, 12.5),
                            show=False, set_xticks_fmt_g=True, set_yticks_fmt_g=True,
                            hists_ylog=False, darkcolor='red', lightcolor='salmon'):
        ''' Returns a list with the 3 axes: dot-plot, upper hist, right hist.
        TODO: Write better description
        '''
        # Create axes
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, y=.99, weight='bold')
        ax_plot = fig.add_axes([.05, .05, .6, .6])
        ax_hist_up = fig.add_axes([.05, .66, .6, .3])
        ax_hist_right = fig.add_axes([.66, .05, .3, .6])

        # Plot stuff
        ### dot plot
        ax_plot.plot(xs, ys, c=darkcolor, marker='o', lw=0)
        ax_plot.set_xlabel(xlabel)
        if set_xticks_fmt_g:
            ax_plot.set_xticklabels(map(fmt_g, ax_plot.get_xticks()))
        if set_yticks_fmt_g:
            ax_plot.set_yticklabels(map(fmt_g, ax_plot.get_yticks()), rotation=90)
        ax_plot.set_ylabel(ylabel)
        ax_plot.grid(grid)

        ### Upper histogram
        m, M = np.floor(xs.min()), np.ceil(xs.max())
        Plotter.plot_pdf_and_cdf(xs, np.linspace(m, M, min(150, max(21, 2*(M-m)+1))), lightcolor, darkcolor, ax=ax_hist_up, grid=True,
                                 ylog=hists_ylog)
        ### Right histogram
        m, M = np.floor(ys.min()), np.ceil(ys.max())
        Plotter.plot_pdf_and_cdf(ys, np.linspace(m, M, min(150, max(21, 2*(M-m)+1))), lightcolor, darkcolor, ax=ax_hist_right, grid=True,
                                 horizontal=True, ylog=hists_ylog)
        ax_hist_up.set_xlim(ax_plot.get_xlim())
        ax_hist_right.set_ylim(ax_plot.get_ylim())
        ax_hist_right.set_xticks(np.linspace(0, 100, 6))
        ax_hist_right.get_xticklabels()[0].set_ha('left')

        if show:
            plt.show()
        else:
            return fig, [ax_plot, ax_hist_up, ax_hist_right]

    @staticmethod
    def plot_pdf_and_cdf(xs, bins, color, cdf_color, xlabel='', ax=None, horizontal=False,
                         grid=True, remove_ticks=True, xlim=None, ylog=False, xticks=None):
        if ax is None:
            ax = plt.gca()
        weights = 100*np.ones(len(xs)) / len(xs)
        xs = np.asarray(xs)
        assert weights.shape == xs.shape
        ns, xs_hist = ax.hist(xs, bins=bins, color=color, label='pdf', weights=weights,
                              orientation='horizontal' if horizontal else 'vertical')[:2]
        fmt_fn = fmt_g
        if ylog:
            fmt_fn = fmt_e
            if horizontal:
                ax.set_xscale('log')
            else:
                ax.set_yscale('log')
        if xlim is not None:
            if horizontal:
                ax.set_ylim(xlim)
            else:
                ax.set_xlim(xlim)
        ylabel = 'Frequency (%)'
        if horizontal:
            xlabel, ylabel = ylabel, xlabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        xs_hist = xs_hist[1:]
        if horizontal:
            ax.plot(np.cumsum(ns), xs_hist, ls=':', lw=2, color=cdf_color, label='cdf')
            ax.set_xlim((0, 100))
            if remove_ticks:
                ax.set_yticklabels([''] * len(ax.get_yticks()))
            else:
                ax.set_yticklabels(map(fmt_fn, ax.get_yticks()))
        else:
            ax.plot(xs_hist, np.cumsum(ns), ls=':', lw=2, color=cdf_color, label='cdf')
            ax.set_ylim((0, 100))
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
            if remove_ticks:
                ax.set_xticklabels([''] * len(ax.get_xticks()))
            else:
                if xticks is not None:
                    if xticks == 'bins':
                        ax.set_xticks(bins)
                    else:
                        ax.set_xticks(xticks)
                ax.set_xticklabels(map(fmt_g, ax.get_xticks()))
        #ax.legend(loc='lower right' if horizontal else 'upper left')
        ax.legend(loc='best')
        ax.grid(grid)

    @staticmethod
    def compare_gene_mappings(integrator, prop_depth=0, leaves_only=False):
        Plotter._set_plot_dir(integrator)
        integrator.propagate_genes(prop_depth)
        prop_depth = integrator.propagation_depth
        hpo2genes_cup = integrator.get_hpo2genes('union')
        hpo2genes_cap = integrator.get_hpo2genes('intersection')
        common_terms = set(hpo2genes_cup.keys()) & set(hpo2genes_cap.keys())
        assert hpo2genes_cap != hpo2genes_cup
        _max = max(map(len, hpo2genes_cup.values()))
        print('([{}])'.format(_max), [t for t, v in hpo2genes_cup.items() if len(v) == _max][0])
        if leaves_only:
            common_terms &= set(integrator.iter_leaves(1))
        x, y, depths = map(lambda l: np.asarray(l, dtype=np.int), zip(*[
            (len(hpo2genes_cup[t]), len(hpo2genes_cap[t]), integrator.get_term_depth(t)) \
            for t in common_terms]))
        r_P = np.corrcoef(x, y)[0, 1]
        r_S = stats.spearmanr(x, y)
        print('Pearson\'s r:', r_P)
        print('Spearman\'s r:', r_S)
        print('\t', np.unique(x), np.unique(y))

        #fig = plt.figure(figsize=(8, 8))
        #ax = fig.add_subplot(111)
        #for depth in np.unique(depths):
        #    if (depths == depth).sum() == 0:
        #        continue
        #    #col = [cm.jet(depth/depths.max())] * (depths == depth).sum()
        #    col = 'b'
        #    ax.scatter(x[depths == depth], y[depths == depth], c=col,
        #               edgecolors=col, label='HPO terms [depth {:d}]'.format(depth))
        fig, axes = Plotter.dot_plot_with_hists(
            x, y, xlabel='Nb of genes associated with HPO terms ($\gamma_\cup$)', ylabel='Nb of genes associated with HPO terms ($\gamma_\cap$)',
            title='Comparison of the number of genes in $\gamma_\cup$ vs $\gamma_\cap$', grid=True, figsize=(6, 6), show=False, darkcolor='dodgerblue', lightcolor='skyblue'
        )
        ax = axes[0]
        M = ax.get_xticks()[-1]
        if M < 1000:
            ticks = ax.get_xticks()[ax.get_xticks() >= 0]
        else:
            ticks = ax.get_xticks()[ax.get_xticks() >= 0][::2]
        ax.plot([0, M], [0, M], 'k-.')
        ax.set_xlim([0, M])
        ax.set_ylim([0, M])
        ax.set_xticks(ticks)
        ax.set_xticklabels(map(fmt_g, ax.get_xticks()))
        ax.get_xticklabels()[-1].set_ha('right')
        ax.set_yticks(ax.get_xticks())
        ax.set_yticklabels(map(fmt_g, ax.get_yticks()))
        ax.get_yticklabels()[-1].set_va('top')
        axes[1].set_xlim(0, M)
        axes[2].set_ylim(0, M)
        axes[1].set_xticks(ticks)
        axes[2].set_yticks(ticks)
        ax.scatter([], [], label='r_P = {:1.2f}\nr_S = {:1.2f}'.format(r_P, r_S[0]), c='dodgerblue')
        ax.legend(loc='best')
        Plotter.save_fig(fig, 'compare_gene_mappings_size_{}{}.eps'.format(prop_depth, '_leaves' if leaves_only else ''))

    @staticmethod
    def plot_hpo2omim(integrator):
        Plotter._set_plot_dir(integrator)
        hpo2omim = integrator.hpo2omim
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        lengths = np.array(list(map(len, hpo2omim.values())))
        N = 50
        M = N * (lengths.max() // N + 1)
        ax.hist(lengths, bins=np.linspace(0, M+1, N+1), weights=100*np.ones(lengths.size)/lengths.size)
        ax.set_xlabel('Number of OMIM associations')
        ax.set_ylabel('Frequency (%)')
        ax.set_yscale('log')
        ax.set_title('Distribution of the number of\nOMIM associations per HPO term')
        ax.grid(True)
        Plotter.save_fig(fig, 'distribution.nb.OMIM.associations.eps')
        omim2hpo = integrator.omim2hpo
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        lengths = np.array(list(map(len, omim2hpo.values())))
        N = 50
        M = N * (lengths.max() // N + 1)
        ax.hist(lengths, bins=np.linspace(0, M+1, N+1), weights=100*np.ones(lengths.size)/lengths.size)
        ax.set_xlabel('Number of HPO associations')
        ax.set_ylabel('Frequency (%)')
        ax.set_yscale('log')
        ax.set_title('Distribution of the number of\nHPO associations per OMIM disease')
        ax.grid(True)
        Plotter.save_fig(fig, 'distribution.nb.HPO.associations.eps')

    @staticmethod
    def plot_density(integrator, gene_mapping):
        Plotter._set_plot_dir(integrator)
        disease2genes = integrator.get_hpo2genes(gene_mapping)
        interactome = integrator.interactome
        interactome.load_density_cache()
        ps = list()
        for genes in disease2genes.values():
            genes &= interactome.genes
            if not genes:
                continue
            size = len(genes)
            densities = interactome.density_cache[size]
            density_genes = interactome.get_genes_density(interactome.verts_id(genes))
            ps.append((densities > density_genes).sum() / len(densities))
        ps = np.asarray(ps)
        ps = np.log10(ps[ps > 0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([np.log10(.05)]*2, [0, 100], 'k-.', label=r'$p = .05$')
        Plotter.plot_pdf_and_cdf(ps, 20, 'salmon', 'r', 'log10(p)', ax=ax, remove_ticks=False)
        Plotter.save_fig(fig, 'density.eps')

    @staticmethod
    def plot_iso_entropy(integrator):
        for gene_mapping in ['intersection', 'union']:
            Plotter._plot_iso_entropy(integrator, gene_mapping)

    @staticmethod
    def _plot_iso_entropy(integrator, gene_mapping):
        Plotter._set_plot_dir(integrator)
        hs, (H, n_classes) = IO.load_entropy(integrator.interactome, gene_mapping)
        hs, ns = map(np.asarray, zip(*hs))
        cheb_p_H = (hs.std() / (H - hs.mean()))**2
        cheb_p_n = (ns.std() / (n_classes - ns.mean()))**2
        p_emp_H = (hs >= H).sum() / len(hs)
        p_emp_n = (ns >= n_classes).sum() / len(ns)
        fig = plt.figure(figsize=(13, 7))
        #---
        ax = fig.add_subplot(121)
        ax.hist(hs, bins=20, label='Random', weights=100*np.ones(len(hs))/len(hs))
        ylim = ax.get_ylim()
        ax.plot([H]*2, ylim, 'r-', lw=2, label='Observed')
        ax.set_ylim(ylim)
        ax.set_xlabel('Entropy (H)')
        xticks = list(ax.get_xticks())
        xticks.append(H)
        xticks.sort()
        ax.set_xticks(xticks)
        xticklabels = [('{:<.' + str(4 if x == H else 2) + 'f}').format(x) for x in xticks]
        new_tick = ax.get_xticklabels()[xticks.index(H)]
        new_tick.set_rotation(90)
        new_tick.set_color('red')
        if H < .4:
            xticklabels[xticks.index(H)] += '   '
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel('Frequency (%)')
        ax.legend(loc='upper left')
        ax.set_title('Distribution of H under uniform sampling\n' + r'($p_{cheb} \leq $' + ('{:.2e}; '.format(cheb_p_H)) + r'$p_{emp} = $' + '{:.2e})'.format(p_emp_H))
        #---
        ax = fig.add_subplot(122)
        ax.hist(ns, bins=np.arange(ns.min(), ns.max()+1)-.5, label='Random', weights=100*np.ones(len(ns))/len(ns))
        NB_TICKS = 20
        m, M = min(ns.min(), n_classes), max(ns.max(), n_classes)
        if M-m <= NB_TICKS:
            xticks = np.arange(m, M+1)
            if n_classes in xticks:
                xticks = np.delete(xticks, np.where(xticks == n_classes)[0][0])
            ax.set_xticks(xticks)
        ylim = ax.get_ylim()
        ax.plot([n_classes]*2, ylim, 'r-', lw=2, label='Observed')
        ax.set_ylim(ylim)
        ax.set_xlabel('Number of isomorphism classes (n)')
        xticks = list(ax.get_xticks())
        xticks.append(n_classes)
        xticks.sort()
        ax.set_xticks(xticks)
        new_tick = ax.get_xticklabels()[xticks.index(n_classes)]
        new_tick.set_rotation(90)
        new_tick.set_color('red')
        if M-m > NB_TICKS:
            ticks = list(map(lambda tick: '{:d}'.format(int(tick)), ax.get_xticks()))
            ticks[xticks.index(n_classes)] += '   '
            ax.set_xticklabels(ticks)
        ax.set_ylabel('Frequency (%)')
        ax.legend(loc='upper left')
        ax.set_title('Distribution of #classes under uniform sampling\n' + r'($p_{cheb} \leq$ ' + \
                     ('{:.2e}; '.format(cheb_p_n)) + r'$p_{emp} =$ ' + ('{:.2e})'.format(p_emp_n))
        )
        fig.suptitle(
            'Distribution of isomorphism behaviour under uniform sampling in the interactome  [{}]'.format(gene_mapping)
        )
        #plt.gcf().set_size_inches(18, 10)
        Plotter.save_fig(fig, 'entropy.{}.eps'.format(gene_mapping))
        # Entropy values details
        print('Number of samples:', len(hs))
        print('Random Entropy  min/max: {:.4f}/{:.4f}'.format(np.min(hs), np.max(hs)))
        print('               mean/std: {:.4f}/{:.4e}'.format(np.mean(hs), np.std(hs)))
        print('ns              min/max: {:4d}/{:4d}'.format(ns.min(), ns.max()))
        print('Sapiro p-values: {:.3e}          {:.3e}' \
              .format(stats.shapiro(hs)[1], stats.shapiro(ns)[1]))
        print('z-scores:        {:.3e}          {:.3e}' \
              .format((H - hs.mean()) / hs.std(), (n_classes - ns.mean()) / ns.std()))
        print('Chebyshev (♥):   {:.3e}          {:.3e}' \
              .format(cheb_p_H, cheb_p_n))
        print('Empirical ps :   {:.3e}          {:.3e}' \
              .format(p_emp_H, p_emp_n))

    @staticmethod
    def plot_relation_degree_to_nb_omim_associations(integrator):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        gene2omim = integrator.get_gene2omim()
        xs, ys = list(), list()
        for gene, phenotypes in gene2omim.items():
            x = interactome.get_gene_degree(gene)
            if x is not None:
                xs.append(x)
                ys.append(len(phenotypes))
        fig, _ = Plotter.dot_plot_with_hists(
            xs, ys, 'Gene degree', '# associated OMIM phenotypes', 'title', grid=True,
            show=False, set_xticks_fmt_g=False, set_yticks_fmt_g=False
        )
        Plotter.save_fig(fig, 'gene_degree_to_nb_omim_associations.eps')

    @staticmethod
    def plot_relation_degree_to_nb_hpo_associations(integrator):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        for mapping in ('intersection', 'union'):
            gene2hpo = integrator.get_gene2hpo()
            xs, ys = list(), list()
            for gene, terms in gene2hpo.items():
                x = interactome.get_gene_degree(gene)
                if x is not None:
                    xs.append(x)
                    ys.append(len(terms))
            fig, _ = Plotter.dot_plot_with_hists(
                xs, ys, 'Gene degree', '# associated HPO terms (' + mapping + ')',
                'title', grid=True, show=False, set_xticks_fmt_g=False, set_yticks_fmt_g=False
            )
            Plotter.save_fig(fig, 'gene_degree_to_nb_hpo_terms (' + mapping + ').eps')

    @staticmethod
    def plot_pathogenic_topologies(integrator):
        size = 5
        Plotter._set_plot_dir(integrator)
        iso_counts = IO.load_topology_analysis(integrator, size)
        keys = iso_counts.keys
        mappings = iso_counts.values
        gene_subset_counts = [sum(map(len, mappings[i].values())) for i in range(len(mappings))]
        s = sum(gene_subset_counts)
        gene_subset_counts = [(i, x/s, x) for (i, x) in enumerate(gene_subset_counts)]
        gene_subset_counts.sort(key=lambda t: t[1], reverse=True)
        print(gene_subset_counts)
        print(len(gene_subset_counts))
        counter = 0
        s = 0
        while counter < len(gene_subset_counts) and s < .9:
            s += gene_subset_counts[counter][1]
            counter += 1
        print(counter, s)
