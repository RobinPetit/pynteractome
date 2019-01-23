import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import obonet

HPO_PATH = '../data/hpo/hpo.obo'

def extract_subtree(T, root, extract_parents=False):
    keys = {root}
    # get children
    queue = [root]
    while queue:
        current_node = queue.pop()
        for parent in set(T[current_node]) - keys:
            queue.append(parent)
            keys.add(parent)
            #print('Adding term {} (child of {})'.format(parent, current_node))
    if not extract_parents:
        return list(keys)
    # get parents
    all_nodes = list(sorted(T.node.keys()))
    all_nodes.remove(root)
    has_added_nodes = True
    while has_added_nodes:
        has_added_nodes = False
        to_remove = list()
        for node in all_nodes:
            intersection = keys & set(T[node])
            if intersection:
                to_remove.append(node)
                keys.add(node)
                has_added_nodes = True
        for node in to_remove:
            all_nodes.remove(node)
    return list(keys)

def draw_graph(G, nodes_blue=None):
    pos = graphviz_layout(G, prog='dot')
    if nodes_blue is None:
        node_color = 'r'
    else:
        node_color = ['b' if node in nodes_blue else 'r' for node in G]
    nx.draw(G, pos=pos, with_labels=True, node_size=500, alpha=.5,
            node_color=node_color, arrows=False)

def get_subtree(G, roots, parents=False):
    if not hasattr(roots, '__iter__'):
        roots = [roots]
    all_nodes = set()
    for root in roots:
        all_nodes |= set(extract_subtree(G, root, parents))
    subtree = nx.subgraph(G, all_nodes)
    return subtree

def draw_subtree(G, roots, parents=False):
    if not hasattr(roots, '__iter__'):
        roots = [roots]
    draw_graph(get_subtree(G, roots, parents), roots)

def get_hpo(data=False):
    graph = obonet.read_obo(HPO_PATH).reverse()
    nx.relabel_nodes(
        graph,
        {original: int(original[len('HP:'):]) for original in graph},
        copy=False
    )
    if data:
        return graph
    # Make a copy to ignore data (avoid errors from `dot` when plotting graph...)
    ontology_copy = type(graph)()
    for node in graph.nodes(data=False):
        ontology_copy.add_node(node)
    for edge in graph.edges(data=False):
        ontology_copy.add_edge(*edge)
    return ontology_copy

def load_abnormal_phenotypes():
    abnormal_phenotype_hpo_id = 118
    ret = get_subtree(
        get_hpo(),
        abnormal_phenotype_hpo_id,
        parents=False
    )
    assert 1 not in ret
    return ret

def main():
    graph = load_abnormal_phenotypes()
    #draw_subtree(graph, [7068], True)
    draw_subtree(graph, [1195], True)
    plt.show()

if __name__ == '__main__':
    main()
