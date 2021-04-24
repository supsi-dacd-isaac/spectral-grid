import networkx as nx
import numpy as np
from mbg import core_partitioning
from mbg.auxiliary_functions import random_conn_graph_2
import argparse
import scipy.io as sio

WL = 'nweight'
EL = 'eweight'

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-type')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()

    if args.type == 'tree':
        cg = nx.random_tree(200)
    elif args.type == 'rand':
        cg = random_conn_graph_2(200, 0.1)
    elif args.type == 'grid':
        cg = nx.generators.grid_2d_graph(20, 20)
    elif args.type == 'comp':
        cg = nx.complete_graph(80)
    else:
        raise ValueError('test type unknown')

    for n in cg.nodes:
        cg.nodes[n][WL] = np.random.rand()

    for e in cg.edges:
        cg.edges[e][EL] = np.random.rand()

    if args.type == 'grid':
        pos = {}
        for n in cg.nodes:
            pos[n] = n
    else:
        pos = nx.spring_layout(cg, weight=EL)

    tot_weight = sum(nx.get_node_attributes(cg, WL).values())
    c, sol = core_partitioning(cg,
                               parts=[.6*tot_weight, .2*tot_weight, .2*tot_weight],
                               aggressiveness=0.1,
                               edge_weight_key=EL,
                               node_weight_key=WL,
                               imbalance_tol=1e-12,
                               maxiter=150,
                               seed=None,
                               do_plots=args.plot,
                               laplacian_mode='norm',
                               graph_plot_pos=pos,
                               verbosity='full')

    if args.save:
        ih = sol['imbalance_history']
        cv = sol['cut_value_history']
        sio.savemat(f'{args.type}.mat', {'imb': ih, 'cut': cv, 'tot': tot_weight})
