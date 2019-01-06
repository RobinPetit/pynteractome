import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import numpy as np
from scipy import stats
# local
from pynteractome.IO import IO
from pynteractome.utils import extract_triangular, fmt_g, fmt_e
from pynteractome.warning import warning

#rc('text', usetex=True)  # Activate LaTeX rendering
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
        path = plot_dir + filename
        ridx = path.rfind('.')
        if ridx > 0:
            ext = path[ridx+1:]
            if ext not in AVAILABLE_FORMATS:
                warning('Unknown format: "{}". Setting default format ("{}").' \
                        .format(ext, AVAILABLE_FORMATS[0]))
            path += '.' + AVAILABLE_FORMATS[0]
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def loc_hpo(integrator):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        for depth in range(integrator.get_hpo_depth()):
            integrator.propagate_genes(depth)
            zs = {term: interactome.get_lcc_score(genes, 0, shapiro=True) \
                  for (term, genes) in integrator.get_hpo2genes().items()}
            Plotter._loc_hpo(integrator, zs, depth)

    @staticmethod
    def _loc_hpo(integrator, zs, prop_depth):
        Plotter._set_plot_dir(integrator)
        integrator.propagate_genes(prop_depth)
        interactome = integrator.interactome
        xs, ys = list(), list()
        are_normal = list()
        empirical_ps = list()
        for term, (z_score, empirical_p, is_normal) in zs.items():
            if z_score is not None:
                z = float(z_score)
                genes = integrator.get_hpo2genes()[term] & interactome.genes
                if len(genes) >= 10:
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
            'Significance via $p$ or $z$ (HPO terms)',
            'barplot.significance.hpo.{}.eps'.format(prop_depth)
        )
        a = np.where(np.logical_and(ys >= 1.65, empirical_ps >= .05))[0]
        print(len(set(a)),
              'terms have significant z but non-significant p')
        print(len(set(np.where(np.logical_and(ys < 1.65, empirical_ps < .05))[0])),
              'terms have non-significant z but significant p')
        if prop_depth == 0:
            title = 'Significance of |LCC| (non-propagated)'
        elif prop_depth == integrator.get_hpo_depth()-1:
            title = 'Significance of |LCC| (up-propagated)'
        else:
            title = 'Significance of |LCC| (up-propagated by {})'.format(prop_depth)
        empirical_ps[empirical_ps < 1e-10] = 1e-10
        empirical_ps = np.log10(empirical_ps)
        path = 'loc/prop.depth.{}.z.eps'.format(prop_depth)
        Plotter._plot_loc_zs(xs, ys, title, path, are_normal)
        path = 'loc/prop.depth.{}.empirical.p.eps'.format(prop_depth)
        Plotter.plot_z_vs_empirical_p(ys, empirical_ps, title, path)
        path = 'loc/prop.depth.{}.p.eps'.format(prop_depth)
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
            if not genes:
                continue
            z, empirical_p, shapiro_p = interactome.get_lcc_score(genes, 0, shapiro=True)
            if z is None:
                continue
            lcc = interactome.get_genes_lcc_size(interactome.verts_id(genes))
            rel_size = lcc / len(genes)
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
            xs, zs, 'Relative size: |LCC|/|S|', 'z-score', title
        )
        ax = axes[0]
        ax.plot([0, 1], [1.65]*2, 'k-.')
        ax.plot([0, 1], [-1.65]*2, 'k-.')
        Plotter.save_fig(fig, path)

    @staticmethod
    def plot_z_vs_empirical_p(zs, ps, title, path):
        fig, axes = Plotter.dot_plot_with_hists(zs, ps, 'z-score', 'log10(Empirical p)', title)
        ax = axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(xlim, [np.log10(.05)]*2, 'k-.')
        ax.plot([1.65, 1.65], ylim, 'k-.')
        Plotter.save_fig(fig, path)

    @staticmethod
    def _plot_loc_ps(xs, empirical_ps, title, path, are_normal):
        print('According to Shapiro-Wilk: {}/{} are normal'.format(are_normal.sum(), len(are_normal)))
        print('{}/{} are significant'.format((empirical_ps < .05).sum(), len(empirical_ps)))
        fig, axes = Plotter.dot_plot_with_hists(
            xs, empirical_ps, 'Relative size: |LCC|/|S|', 'log10(Empirical p)', title
        )
        ax = axes[0]
        xlim = ax.get_xlim()
        ax.plot(xlim, [np.log10(.05)]*2, 'k-.')
        Plotter.save_fig(fig, path)

    @staticmethod
    def significance_bar_plot(zs, empirical_ps, title, path):
        non_significant = np.where(np.logical_and(zs <= 1.65, empirical_ps >= .05))[0]
        z_significant = np.where(np.logical_and(zs > 1.65, empirical_ps >= .05))[0]
        p_significant = np.where(np.logical_and(zs <= 1.65, empirical_ps < .05))[0]
        significant = np.where(np.logical_and(zs > 1.65, empirical_ps < .05))[0]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        lengths = np.asarray(
            [len(non_significant), len(z_significant), len(p_significant), len(significant)]
        )
        lengths = 100*lengths / lengths.sum()
        ax.bar(
            np.arange(len(lengths)), lengths, align='center', color='salmon',
            tick_label=(
                r'$z \leq 1.65$ and $p \geq .05$',
                r'$z > 1.65$ and $p \geq .05$',
                r'$z \leq 1.65$ and $p < .05$',
                r'$z > 1.65$ and p < .05'
            )
        )
        for x, y in zip(ax.get_xticks(), lengths):
            ax.text(x-.1, y+5, '{:.2f}%'.format(y), weight='bold')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Frequency')
        ax.grid(True)
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
    def gamma_density(integrator):
        Plotter._set_plot_dir(integrator)
        non_empty_links = list()
        term2genes = integrator.get_hpo2genes()
        for term in term2genes:
            genes = term2genes[term] & integrator.interactome.genes
            if len(genes) > 20:
                non_empty_links.append(len(genes))
        print(np.max(non_empty_links))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(non_empty_links, 200)
        Plotter.save_fig(fig, 'gamma_density;eps')

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
        Plotter.save_fig(fig, path)

    @staticmethod
    def plot_deg_power_law(integrator):
        Plotter._set_plot_dir(integrator)
        interactome = integrator.interactome
        degs = interactome.G.get_out_degrees(np.arange(interactome.G.num_vertices()))
        degrees = np.arange(degs.max())+1
        counts = np.zeros(degrees.shape, dtype=np.float)
        for d in degs:
            counts[int(d)-1] += 1
        counts /= degrees.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(degrees[counts != 0], counts[counts != 0], 'ro', mec='red')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree distribution of the interactome')
        Plotter.save_fig(fig, 'deg_power_law.eps')

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
        ax = fig.add_subplot(111)
        all_values = [
            [len(hpo2omim[term]) for term in integrator.order_n_ontology(n) if term in hpo2omim] \
                for n in range(integrator.get_hpo_depth())
        ]
        indices = list()
        vs = list()
        for i, v in enumerate(all_values):
            if len(v) != 0:
                indices.append(i+1)
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
                indices.append(i+1)
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
                            hists_ylog=False):
        ''' Returns a list with the 3 axes: dot-plot, upper hist, right hist.
        '''
        # Create axes
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, y=.99, weight='bold')
        ax_plot = fig.add_axes([.05, .05, .6, .6])
        ax_hist_up = fig.add_axes([.05, .66, .6, .3])
        ax_hist_right = fig.add_axes([.66, .05, .3, .6])

        # Plot stuff
        ### dot plot
        ax_plot.plot(xs, ys, 'ro')
        ax_plot.set_xlabel(xlabel)
        if set_xticks_fmt_g:
            ax_plot.set_xticklabels(map(fmt_g, ax_plot.get_xticks()))
        if set_yticks_fmt_g:
            ax_plot.set_yticklabels(map(fmt_g, ax_plot.get_yticks()), rotation=90)
        ax_plot.set_ylabel(ylabel)
        ax_plot.grid(grid)

        ### Upper histogram
        Plotter.plot_pdf_and_cdf(xs, 20, 'salmon', 'r', ax=ax_hist_up, grid=True,
                                 ylog=hists_ylog)
        ### Right histogram
        Plotter.plot_pdf_and_cdf(ys, 20, 'salmon', 'r', ax=ax_hist_right, grid=True,
                                 horizontal=True, ylog=hists_ylog)

        if show:
            plt.show()
        else:
            return fig, [ax_plot, ax_hist_up, ax_hist_right]

    @staticmethod
    def plot_pdf_and_cdf(xs, bins, color, cdf_color, xlabel='', ax=None, horizontal=False,
                         grid=True, remove_ticks=True, xlim=None, ylog=False):
        if ax is None:
            ax = plt.gca()
        weights = 100*np.ones(len(xs)) / len(xs)
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
                ax.set_xticklabels(map(fmt_g, ax.get_xticks()))
        ax.legend(loc='best')
        ax.grid(grid)

    @staticmethod
    def compare_gene_mappings(integrator):
        Plotter._set_plot_dir(integrator)
        hpo2genes_cup = integrator.get_hpo2genes('union')
        hpo2genes_cap = integrator.get_hpo2genes('intersection')
        common_terms = set(hpo2genes_cup.keys()) & set(hpo2genes_cap.keys())
        x, y, depths = map(lambda l: np.asarray(l, dtype=np.int), zip(*[
            (len(hpo2genes_cup[t]), len(hpo2genes_cap[t]), integrator.get_term_depth(t)) \
            for t in common_terms]))
        print('Pearson\'s r:', np.corrcoef(x, y)[0, 0])
        print('Spearman\'s r:', stats.spearmanr(x, y))

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        for depth in np.unique(depths):
            if (depths == depth).sum() == 0:
                continue
            col = [cm.jet(depth/depths.max())] * (depths == depth).sum()
            ax.scatter(x[depths == depth], y[depths == depth], c=col,
                       edgecolors=col, label='HPO terms [depth {:d}]'.format(depth))
        xlim = (0, ax.get_xlim()[1])
        ax.plot(xlim, xlim, 'k-.')
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel('Nb of genes associated w/ HPO terms [union]')
        ax.set_ylabel('Nb of genes associated w/ HPO terms [intersection]')
        ax.set_title('Comparison # genes union-mapping vs intersection-mapping')
        plt.grid(True)
        plt.legend(loc='upper left')
        Plotter.save_fig(fig, 'compare_gene_mappings.eps')

    @staticmethod
    def plot_density(integrator):
        Plotter._set_plot_dir(integrator)
        disease2genes = integrator.get_hpo2genes()
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
        Plotter._set_plot_dir(integrator)
        hs, (H, n_classes) = IO.load_entropy(integrator.interactome)
        hs, ns = map(np.asarray, zip(*hs))
        fig = plt.figure()
        #---
        ax = fig.add_subplot(121)
        ax.hist(hs, bins=20, label='Random', weights=np.ones(len(hs))/len(hs))
        ylim = ax.get_ylim()
        ax.plot([H]*2, ylim, 'r-', lw=2, label='Observed')
        ax.set_ylim(ylim)
        ax.set_xlabel('Entropy (H)')
        xticks = list(ax.get_xticks())
        xticks.append(H)
        xticks.sort()
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [('{:<.' + str(4 if x == H else 2) + 'f}').format(x) for x in xticks]
        )
        new_tick = ax.get_xticklabels()[xticks.index(H)]
        new_tick.set_rotation(90)
        new_tick.set_color('red')
        ax.set_ylabel('P(H)')
        ax.legend(loc='upper center')
        ax.set_title('Distribution of H under uniform sampling')
        #---
        ax = fig.add_subplot(122)
        ax.hist(ns, bins=20, label='Random', weights=np.ones(len(ns))/len(ns))
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
        ax.set_ylabel('P(n)')
        ax.legend(loc='upper center')
        ax.set_title('Distribution of #classes under uniform sampling')
        fig.suptitle(
            'Distribution of isomorphism behaviour under uniform sampling in the interactome'
        )
        plt.gcf().set_size_inches(18, 10)
        Plotter.save_fig(fig, 'entropy.eps')
        # Entropy values details
        print('Number of samples:', len(hs))
        print('Random Entropy  min/max: {:.4f}/{:.4f}'.format(np.min(hs), np.max(hs)))
        print('               mean/std: {:.4f}/{:.4e}'.format(np.mean(hs), np.std(hs)))
        print('Sapiro p-values: {:.3e}          {:.3e}' \
              .format(stats.shapiro(hs)[1], stats.shapiro(ns)[1]))
        print('z-scores:        {:.3e}          {:.3e}' \
              .format((H - hs.mean()) / hs.std(), (n_classes - ns.mean()) / ns.std()))
