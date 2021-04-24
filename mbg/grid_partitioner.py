import networkx as nx

from .spectral import core_partitioning

INV_WEIGHT = 'inv_weight'


def partition_grid(graph, parts, edge_weight_key, node_weight_key, partitioning_kwargs):

    # inverse mapping virtual length
    maxi = 0
    for e in graph.edges:
        if graph.edges[e][edge_weight_key] > 0:
            graph.edges[e][INV_WEIGHT] = 1/graph.edges[e][edge_weight_key]
            if 1/graph.edges[e][edge_weight_key] > maxi:
                maxi = 1/graph.edges[e][edge_weight_key]
        else:
            pass

    for e in graph.edges:
        if graph.edges[e][edge_weight_key] > 0:
            graph.edges[e][INV_WEIGHT] = graph.edges[e][INV_WEIGHT] / maxi
        else:
            graph.edges[e][INV_WEIGHT] = 2

    kwargs = dict(
        aggressiveness=0.05,
        imbalance_tol=1e-1,
        maxiter=150,
        laplacian_mode='norm',  # 'norm', 'weight', 'ratio'
        seed=None,
        verbosity='full',  # 'none', 'full', 'progress'
        do_plots=False,
        graph_plot_pos=dict(nx.get_node_attributes(graph, 'pos'))
    )

    kwargs.update(partitioning_kwargs)

    best, sol = core_partitioning(graph,
                                  parts,
                                  INV_WEIGHT,
                                  node_weight_key,
                                  **kwargs)

    labels = set(best.values())
    cl_lists = [set([k for k, v in best.items() if v == l]) for l in labels]
    return cl_lists
