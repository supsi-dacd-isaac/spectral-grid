import logging
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pprint import pformat

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from sklearn.cluster import k_means
from tqdm import tqdm

matplotlib.use('Qt5Agg')


def core_partitioning(graph,
                      parts,
                      edge_weight_key,
                      node_weight_key,

                      aggressiveness=1e-3,
                      imbalance_tol=1e-1,
                      maxiter=100,
                      laplacian_mode='norm',  # 'norm', 'weight', 'ratio'
                      seed=None,
                      verbosity='full',  # 'none', 'full', 'progress'
                      do_plots=False,
                      graph_plot_pos=None):

    n_parts = len(parts)
    parts = np.sort(parts)
    work_graph = deepcopy(graph)  # working copy in which the edge weights are modified
    imbalance_history = np.asarray([])
    best_imbalance_history = np.asarray([])
    cut_value_history = np.asarray([])
    solution_history = []
    niter = 0
    best_imbalance = np.Inf
    best_clusters = None
    cut_of_best = None
    tot_weight = sum(nx.get_node_attributes(work_graph, node_weight_key).values())
    tot_weight_norm = tot_weight / n_parts
    if do_plots and graph_plot_pos is None:
        graph_plot_pos = nx.spring_layout(graph)

    # the target_imbalance is calculated as a fraction of the normalized total weight.
    target_imbalance = imbalance_tol * tot_weight_norm

    if verbosity == 'full':
        stream_log_level = logging.INFO
        enable_bar = False
    elif verbosity == 'progress':
        stream_log_level = logging.WARNING
        enable_bar = True
    elif verbosity == 'none':
        stream_log_level = logging.WARNING
        enable_bar = False
    else:
        raise ValueError(f'unknown verbosity setting {verbosity} (none|full|progress)]')

    # set up logger
    logger = logging.getLogger('relax_partition')
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(stream_log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    # kmeans has fixed seed across iterations
    if seed is None:
        seed = np.random.randint(1, 99999999)

    prog_bar = tqdm(disable=not enable_bar, total=maxiter, ascii=True, leave=True)
    target_reached = False

    while True:

        # PHASE 1: ASSESS CURRENT CONDITION ---------------------

        # calculate laplacian

        # note: all matrices are sparse
        A = nx.adjacency_matrix(work_graph, weight=edge_weight_key)
        L0 = nx.laplacian_matrix(work_graph, weight=edge_weight_key)
        D = L0 + A

        # vector of weighted nodes
        node_weights = nx.get_node_attributes(work_graph, node_weight_key)

        if laplacian_mode.startswith('ratio'):
            LM = L0
        elif laplacian_mode.startswith('norm'):
            D12 = D.power(-1/2)
            LM = D12 * L0 * D12
        elif laplacian_mode.startswith('weight'):
            weight_vector = np.asarray([node_weights[n] for n in work_graph.nodes()])
            M = diags(weight_vector)
            M12 = M.power(-1/2)
            LM = M12 * L0 * M12
        else:
            raise ValueError(f'laplacian mode {laplacian_mode} unknown (ratio|norm|weight)')

        # compute eigenvalues
        try:
            eig_values, eig_vectors = eigs(L0, k=n_parts, M=D, which='SM')
        except:
            break
        pU = np.real(eig_vectors)

        # normalization and kmeans
        spu = np.repeat(np.sqrt(np.sum(np.square(pU), axis=1, keepdims=True)), n_parts, axis=1)
        pU = np.divide(pU, spu)
        _centroids, labels, _ = k_means(pU, n_parts, random_state=seed)
        node2label = dict(zip(work_graph.nodes(), labels))  # maps nodes to clusters

        # calculate cluster weights
        ws = defaultdict(lambda: 0)
        for node, cluster in node2label.items():
            ws[cluster] += graph.nodes[node][node_weight_key]
        current_imbalance = np.max(np.abs(np.sort(parts) - np.sort(list(ws.values()))))

        # calculate cluster_sets and current cut value

        # edge weights represent SIMILARITY-AFFINITY. higher value = more affinity.
        # In fact, 0 is the least possible affinity! (no edge)
        # therefore, a good cut value is a LOW cut value.

        cluster_sets = defaultdict(set)
        for node, label in node2label.items():
            cluster_sets[label].add(node)
        # the cut is calculated on the ORIGINAL graph!
        current_cut_value = calc_cut(graph, cluster_sets, edge_weight_key)

        # keep track of the best (least imbalanced) solution
        if current_imbalance < best_imbalance:
            best_imbalance = current_imbalance
            best_clusters = node2label
            cut_of_best = current_cut_value

        # keep track of solution history
        imbalance_history = np.concatenate([imbalance_history, [current_imbalance]])
        best_imbalance_history = np.concatenate([best_imbalance_history, [best_imbalance]])
        solution_history.append(node2label)
        cut_value_history = np.concatenate([cut_value_history, [current_cut_value]])

        # log current state of the solution
        logger.info(f' it: {niter:04}/{maxiter} |  '
                    f'BEST imb:{best_imbalance/tot_weight_norm*100:.3}% @cut:{cut_of_best:.3} |  '
                    f'CURR imb:{current_imbalance/tot_weight_norm*100:.3}% @cut:{current_cut_value:.3}')

        # check break conditions
        niter += 1
        if best_imbalance <= target_imbalance:
            target_reached = True
            break

        if niter >= maxiter:
            break

        # optional plots
        if do_plots:
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(13.5, 8.0)
            plt.ion()  # interactive mode
            plt.clf()

            # imbalance history
            plt.subplot(2, 2, 1)
            plt.title('Max relative cluster imbalance')
            plt.gca().set_ylim(bottom=0)
            plt.plot(imbalance_history/tot_weight_norm, 'b:')
            plt.plot(best_imbalance_history/tot_weight_norm, 'k-')
            plt.plot(np.ones_like(imbalance_history) * imbalance_tol, 'r--')
            plt.legend(['current', 'best', 'target'])

            # cut history
            plt.subplot(2, 2, 3)
            plt.title('Cut value (lower is better)')
            plt.gca().set_xlabel('iteration')
            plt.plot(cut_value_history, 'b-')

            # graph visualization
            plt.subplot(1, 2, 2)
            plt.title('Graph visualization')
            ec = nx.get_edge_attributes(work_graph, edge_weight_key)
            edgelist = list(ec.keys())
            edgecols = list(ec.values())

            nx.draw_networkx(work_graph,
                             with_labels=False,
                             labels=node2label,
                             pos=graph_plot_pos,
                             node_color=list(node2label.values()),
                             node_size=100,
                             edgelist=edgelist,
                             edge_color=edgecols,
                             vmin=min(node2label.values()),
                             vmax=max(node2label.values()))

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        # PHASE 2: PERFORM HEURISTIC UPDATE ------------------

        # determine givers and takers based on cluster weight
        cluster_weights = np.zeros(np.max(labels) + 1)
        for nd, l in node2label.items():
            cluster_weights[l] += node_weights[nd]

        scw = np.sort(cluster_weights)
        sca = np.argsort(cluster_weights)

        sca_i = {i: x for i, x in enumerate(sca)}

        givers = []
        takers = []
        for cursor, label in sca_i.items():
            if scw[cursor] >= parts[cursor]:
                givers.append(label)
            else:
                takers.append(label)

        if len(givers) == 0:
            logger.warning('No givers found')
            break

        labels_catalog = set(labels)
        cluster_pairs = combinations(labels_catalog, 2)

        for l1, l2 in cluster_pairs:

            # relaxation happens only between giver-taker pairs
            if (l1 in takers and l2 in takers) or (l1 in givers and l2 in givers):
                continue

            if l1 in givers:
                giver_label = l1
                taker_label = l2

            else:
                giver_label = l2
                taker_label = l1

            # identify which nodes of the giver cluster lie at the border
            giver_cluster = cluster_sets[giver_label]
            taker_cluster = cluster_sets[taker_label]
            giver_subgraph = nx.subgraph(work_graph, giver_cluster)
            taker_subgraph = nx.subgraph(work_graph, taker_cluster)
            union_subgraph = nx.subgraph(work_graph, giver_cluster.union(taker_cluster))
            edges_giver = set([frozenset(x) for x in giver_subgraph.edges])
            edges_taker = set([frozenset(x) for x in taker_subgraph.edges])
            edges_union = set([frozenset(x) for x in union_subgraph.edges])

            intercluster_edges = edges_union - edges_giver.union(edges_taker)
            border_nodes_both = list(sum([tuple(x) for x in intercluster_edges], ()))
            border_nodes_giver = [x for x in border_nodes_both if x in giver_cluster]

            # if there are no bordering nodes, this cycle is not entered
            for giver_node in border_nodes_giver:
                # for each node at the giver's border, find the neighbors
                nbrs = nx.neighbors(union_subgraph, giver_node)
                for nn in nbrs:
                    if nn in giver_cluster:
                        # if the neighbor is in the same cluster, "loosen" the edge
                        work_graph.edges[(giver_node, nn)][edge_weight_key] = \
                            work_graph.edges[(giver_node, nn)][edge_weight_key] * (1 - aggressiveness)
                    else:
                        # if the neighbor is in the other cluster, "tighten" the edge
                        assert nn in taker_cluster
                        work_graph.edges[(giver_node, nn)][edge_weight_key] = \
                            work_graph.edges[(giver_node, nn)][edge_weight_key] * (1 + aggressiveness)

        prog_bar.update(1)
        prog_bar.desc = f'<BEST imb:{best_imbalance/tot_weight_norm*100:.3}% @cut:{cut_of_best:.3}>  iters'

    if do_plots:
        plt.ioff()
        plt.show()

    # solution info
    succint_sol = {
        'target_reached': target_reached,
        'solution_imbalance': best_imbalance,
        'solution_imbalance_relative': best_imbalance / tot_weight_norm,
        'solution_cut_value': cut_of_best,
        'n_iterations': niter,
    }

    hists = {
        'imbalance_history': imbalance_history,
        'solution_history': solution_history,
        'cut_value_history': cut_value_history
    }

    full_sol = {**succint_sol, **hists}

    logger.info(f'final_solution:\n\n{pformat(succint_sol)}')

    return best_clusters, full_sol


def calc_cut(graph, cluster_sets, edge_weight_key):

    wbase = sum(nx.get_edge_attributes(graph, edge_weight_key).values())
    wsum = 0.0
    for label, c in cluster_sets.items():
        subgraph = nx.subgraph(graph, c)
        wsum += sum(nx.get_edge_attributes(subgraph, edge_weight_key).values())
    return wbase - wsum


if __name__ == '__main__':
    pass