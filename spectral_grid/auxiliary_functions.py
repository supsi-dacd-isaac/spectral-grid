import os
import random
from collections import Counter
from copy import deepcopy
from math import radians as rad

import networkx as nx
import numpy as np
from numpy import isclose, asarray, mean, argsort
from tqdm import tqdm

from .projection import project

R_EARTH = 6378137.0

EXPLORED_POINTS_BREADTH = 4
N_DAISIES = 2

this_dir = os.path.dirname(os.path.realpath(__file__))


def sort_box(box):
    lat_bounds = sorted([box[0], box[2]])
    long_bounds = sorted([box[1], box[3]])

    return lat_bounds[0], long_bounds[0], lat_bounds[1], long_bounds[1]


def calc_length(g, node_couple, pos_key):

    p1 = g.nodes[node_couple[0]][pos_key]
    p2 = g.nodes[node_couple[1]][pos_key]

    return np.sqrt((rad(p1[0]) - rad(p2[0]))**2 + (rad(p1[1]) - rad(p2[1]))**2) * R_EARTH


def calc_length_primitive(lat1, long1, lat2, long2):
    return np.sqrt((rad(lat1) - rad(lat2))**2 + (rad(long1) - rad(long2))**2) * R_EARTH


def _walk2(iterable):
    # "s -> (s0, s1), (s1, s2), (s2, s3), ..."

    itr = iter(iterable)

    a = next(itr)

    while True:
        try:
            b = next(itr)
        except StopIteration:
            break

        yield a, b
        a = b


def way_area(way):

    s = 0
    n_nodes = len(way.nodes)
    for nn in range(n_nodes):
        nn1 = (nn % n_nodes)
        nn2 = ((nn + 1) % n_nodes)
        node1 = way.get_nodes()[nn1]
        node2 = way.get_nodes()[nn2]

        s += (rad(node1.lat) * rad(node2.lon)) - (rad(node2.lat) * rad(node1.lon))

    return abs(s * 0.5 * R_EARTH**2)


def way_center(way):

    ns = way.get_nodes()

    lats = [n.lat for n in ns]
    lons = [n.lon for n in ns]

    return float(mean(lons)), float(mean(lats))


def cluster_centroid(buildings):
    weights = np.asarray([way_area(x) for x in buildings])
    centers = np.asarray([(float(way_center(x)[0]), float(way_center(x)[1])) for x in buildings])
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]

    X = np.sum(np.multiply(xs, weights)) / np.sum(weights)
    Y = np.sum(np.multiply(ys, weights)) / np.sum(weights)

    return X, Y


def node_xy(node):
    return float(node.lon), float(node.lat)


def _k_nearest(k, base_point_coords, other_points_dict):
    names, coords = zip(*other_points_dict.items())
    np_coords = asarray(coords) - base_point_coords
    np_dist = np_coords[:, 0] ** 2 + np_coords[:, 1] ** 2
    classifica = argsort(np_dist)
    return [names[classifica[x]] for x in range(k)]


def _default_phase_choose(gr, proj_node, phases=(1, 2, 3)):

    cc = Counter(dict(zip(phases, [0] * len(phases))))
    node_dist = nx.shortest_path_length(gr, source=proj_node)
    maxdist = max(node_dist.values()) + 1

    for edge in gr.edges:
        if gr.edges[edge]['type'] != 'link':
            continue
        weight = maxdist - mean([node_dist[edge[0]], node_dist[edge[1]]])
        cc.update({gr.edges[edge]['phase']: weight})

    mc = cc.most_common()
    return mc[-1][0]


def random_conn_graph_2(n, p):

    gr = nx.random_tree(n)
    cg = nx.complete_graph(n)

    initial_edges = set(gr.edges)
    possible_edges = set(cg.edges)

    n_possible_edges = n * (n - 1) / 2
    n_target_edges = int(p*n_possible_edges)
    n_initial_edges = gr.number_of_edges()

    edges_to_create = n_target_edges - n_initial_edges

    available_edges = list(possible_edges-initial_edges)
    random.shuffle(available_edges)

    edges_selected = available_edges[0: max(0, int(edges_to_create))]
    gr.add_edges_from(edges_selected)
    assert int(edges_to_create) < 1 or gr.number_of_edges() == n_target_edges
    return gr


def total_connect(gr: nx.Graph,
                  pts: list,
                  middle_projected_node_name: dict,
                  connection_type_weigths: dict,
                  filter_edge_attr={},
                  phase_picker_fn=_default_phase_choose,
                  position_attribute_name='pos',
                  length_attribute_name='virtual_length',
                  rtol_street=1e-7,
                  atol_street=1e-9,
                  logger=None,
                  **edge_props):

    s_gr = deepcopy(gr)

    street_positions_dict = {k: v for k, v in dict(nx.get_node_attributes(s_gr, position_attribute_name)).items() if k not in pts}
    building_positions_dict = {k: v for k, v in dict(nx.get_node_attributes(s_gr, position_attribute_name)).items() if k in pts}
    # positions must only be the streets, because at first we do the best street connections, and towards the end we
    # realize all the daisy chains separately

    for pt_name in tqdm(pts, desc='Connecting buildings to skeleton... ', unit='buildings', disable=logger.level > 10):

        ptpt = asarray(gr.nodes[pt_name][position_attribute_name])

        # here we find a small bunch of edges that are eligible for being the nearest neighbor
        cool_points = _k_nearest(EXPLORED_POINTS_BREADTH, ptpt, street_positions_dict)
        cool_edges = list(s_gr.edges(cool_points))

        for idx, edge in enumerate(cool_edges):
            # hard coded: no attachments are ever made on the main entry line.
            if gr.edges[edge]['type'] == 'entry':
                continue

            # we filter only the edges we want.
            for attr, value in filter_edge_attr.items():
                try:
                    if gr.edges[edge][attr] != value:
                        filtered = True
                        break
                except KeyError:
                    pass
            else:
                filtered = False

            if filtered:
                continue

            point_onedge = project(ptpt, gr.nodes[edge[0]][position_attribute_name],
                                    gr.nodes[edge[1]][position_attribute_name])

            # if the projected point coincides with one of the endpoints we use it as projected name
            if isclose(asarray(point_onedge), gr.nodes[edge[0]][position_attribute_name], rtol=rtol_street, atol=atol_street).all():
                terminal = True
                projected_node_name = edge[0]
            elif isclose(asarray(point_onedge), gr.nodes[edge[1]][position_attribute_name], rtol=rtol_street, atol=atol_street).all():
                terminal = True
                projected_node_name = edge[1]
            else:
                terminal = False
                projected_node_name = middle_projected_node_name[pt_name]

            selected_edge_attributes = gr.edges[edge]

            if not terminal:
                projected_node_name = projected_node_name + '_' + str(idx)

                # adds two new edges with the projected node in the middle
                gr.add_edge(edge[0], projected_node_name, **selected_edge_attributes)
                gr.add_edge(projected_node_name, edge[1], **selected_edge_attributes)

                # inserts position and type of the projected node name
                gr.nodes[projected_node_name][position_attribute_name] = tuple(point_onedge)
                gr.nodes[projected_node_name]['type'] = 'street_link'

            # ok, now we can decide which are the best phases to attach.
            # if we're daisy-chaining an existing link, we must keep the same phase
            if selected_edge_attributes['type'] == 'link':
                edge_props['phase'] = selected_edge_attributes['phase']
            # otherwise, we're free to choose with our phase picking function.
            elif edge_props['phases'] == 1:
                ph = phase_picker_fn(gr, projected_node_name)
                edge_props['phase'] = ph

            gr.add_edge(projected_node_name, pt_name, **edge_props)  # adds the edge to the starting point

    # now we add some (sensible) daisy chain possibilities
    # new_edges = combinations(pts, 2)
    for pt_name in tqdm(pts, desc='Daisy-chaining buildings... ', unit='buildings', disable=logger.level > 10):
        ptpt = asarray(gr.nodes[pt_name][position_attribute_name])
        brothers = _k_nearest(N_DAISIES, ptpt, building_positions_dict)
        for b in brothers:
            gr.add_edge(pt_name, b, type='daisy_chain')

    # now we assign the lengths, which will be the weights for the mst search
    for ed in gr.edges:
        gr.edges[ed][length_attribute_name] = calc_length(gr, ed, 'pos') * connection_type_weigths[gr.edges[ed]['type']]
