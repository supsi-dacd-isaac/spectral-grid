from collections import OrderedDict
from itertools import tee
from types import MethodType
from warnings import simplefilter

import networkx as nx
import numpy as np
from scipy.spatial import distance as spd


R_EARTH = 6378137.0
DEFAULT_CONN_WEIGHTS = {
    'street': 1.1,
    'link': 1.7,
    'daisy_chain': 2.5,
    'entry': 10
}


class SuppressFutureWarnings(object):
    def __enter__(self):
        simplefilter(action='ignore', category=FutureWarning)

    def __exit__(self, exc_type, exc_val, exc_tb):
        simplefilter(action='default', category=FutureWarning)


def treeize(graph: nx.Graph, root_node):

    def leaves(self):
        return [x for x in self.nodes() if self.out_degree(x) == 0]

    if not nx.is_tree(graph):
        raise nx.NotATree('treeize needs a graph with a tree structure')

    assert root_node in graph.nodes

    bunched = [root_node]
    group = [root_node]
    tree = nx.DiGraph()
    tree.leaves = MethodType(leaves, tree)

    while group:
        newgroup = []
        for node in group:
            neighborhood = graph.neighbors(node)
            for neighbor in neighborhood:
                if neighbor not in bunched:
                    tree.add_edge(node, neighbor)
                    bunched.append(neighbor)
                    newgroup.append(neighbor)
        group = newgroup

    return tree


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def sort_box(box):
    lat_bounds = sorted([box[0], box[2]])
    long_bounds = sorted([box[1], box[3]])

    return lat_bounds[0], long_bounds[0], lat_bounds[1], long_bounds[1]


def is_inside(point, box):

    long, lat = point
    sbox = sort_box(box)

    if sbox[0] <= lat <= sbox[2]:
        if sbox[1] <= long <= sbox[3]:
            return True

    return False


def nearest_node(graph, point, filter_key=None, filter_value=None, pos_key='pos'):
    """If all nodes of a graph have a 2-tuple attribute named pos_prop, this
    function finds the one that's nearest to the 2-tuple 'point', interpreting
    the tuples as x-y pairs on a euclidean plane."""
    pdic = OrderedDict(nx.get_node_attributes(graph, pos_key))

    if filter_key is not None:
        filter_dic = nx.get_node_attributes(graph, filter_key)
        filtered_nodes = [k for k, v in filter_dic.items() if v == filter_value]
    else:
        filtered_nodes = [k for k, v in pdic.items()]

    tlist = np.array([v for k, v in pdic.items() if k in filtered_nodes])
    pt = np.asarray(point).reshape(1, -1)
    distances = spd.cdist(tlist, pt)
    mind = int(np.argmin(distances))

    nn_name = list(pdic.keys())[mind]

    return nn_name
