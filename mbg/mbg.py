import json
import logging
import matplotlib.colors as colors
import networkx as nx
import networkx.algorithms.approximation.steinertree as nxs
import os.path
import overpy as oy
import shapely.geometry as shg
import shelve
from copy import deepcopy
from functools import partial
from map2graph._utils import treeize, pairwise
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from numpy import asarray, Inf, mean, asarray
from operator import itemgetter
from overpy import Overpass
from sys import version_info
from time import time

from .auxiliary_functions import this_dir, way_center, way_area, sort_box, _walk2
from .auxiliary_functions import total_connect
from .grid_partitioner import partition_grid
from .map_painting import DEFAULT_GRAPHIC_OPTS, _paint_map

PY2 = version_info[0] == 2
PY3 = version_info[0] == 3

if PY2:
    from urllib2 import urlopen
    from urllib2 import HTTPError
elif PY3:
    from urllib.request import urlopen
    from urllib.error import HTTPError


DEFAULT_CONFIG = dict(
    daisy_chain=True
)


CONN_WEIGHTS = {'street': 0.6,
                'link': 1.2,
                'daisy_chain': 1.5,
                'entry': 100}


class _MapBoxEncoder(json.JSONEncoder):
    def default(self, o):

        if isinstance(o, oy.Way):
            return {'nodes': o.nodes, 'id': o.id}

        if isinstance(o, oy.Node):
            return o.id

        if hasattr(o, 'magnitude'):
            return {'magnitude': o.magnitude, 'unit': o.units.__str__()}


class _MockF:
    def __init__(self, query, f):
        self.code = f.code
        self.query = query

        try:
            self._info = None
            self._header = f.getheader("Content-Type")
        except:
            self._info = f.info()
            self._header = None

    def getheader(self, cty):
        if cty != "Content-Type":
            raise ValueError
        elif self._header is None:
            raise AttributeError
        else:
            return self._header

    def info(self):
        if self._info is None:
            raise AttributeError
        else:
            return self._info

    @staticmethod
    def pretend_to_do_sthg():
        pass


class _SplitQueryOverpass(oy.Overpass):
    # this mod is needed because oy.Result is not picklable and cannot be shelved, although the textual responses from
    # openstreetmaps can. The query function is split in 2 in order to allow retrieval of this intermediate results for
    # i/o purposes.
    # The auxiliary method "query from file" is also added

    def conclude_raw_query(self, response, f):

        if f.code == 200:
            if PY2:
                http_info = f.info()
                content_type = http_info.getheader("content-type")
            else:
                content_type = f.getheader("Content-Type")

            if content_type == "application/json":
                return self.parse_json(response)

            if content_type == "application/osm3s+xml":
                return self.parse_xml(response)

            raise oy.exception.OverpassUnknownContentType(content_type)

        if f.code == 400:
            msgs = []
            for msg in self._regex_extract_error_msg.finditer(response):
                tmp = self._regex_remove_tag.sub(b"", msg.group("msg"))
                try:
                    tmp = tmp.decode("utf-8")
                except UnicodeDecodeError:
                    tmp = repr(tmp)
                msgs.append(tmp)

            raise oy.exception.OverpassBadRequest(
                f.query,
                msgs=msgs
            )

        if f.code == 429:
            raise oy.exception.OverpassTooManyRequests

        if f.code == 504:
            raise oy.exception.OverpassGatewayTimeout

        raise oy.exception.OverpassUnknownHTTPStatusCode(f.code)

    def query_raw(self, query):
        if not isinstance(query, bytes):
            query = query.encode("utf-8")

        try:
            f = urlopen(self.url, query)
        except HTTPError as e:
            f = e

        response = f.read(self.read_chunk_size)
        while True:
            data = f.read(self.read_chunk_size)
            if len(data) == 0:
                break
            response = response + data
        f.close()

        return response, _MockF(query, f)

    def query(self, query):
        # overridden for coherence
        return self.conclude_raw_query(*self.query_raw(query))


class MapBoxGraph:

    def __init__(self, box,
                 config={},
                 log_format='%(asctime)s (MapBoxGraph) %(levelname)s --> %(message)s',
                 log_level=logging.WARNING):
        """Given a box rectangle, returns a nx.Graph that runs into the streets found in that box and has leaves
        corresponding to the centroids of the buildings found in that box. Returns also the buildings and the highways
        as OpenStreetMap ways (paths of points)."""

        super().__init__()
        self.box = sort_box(box)

        # conf
        self.config = DEFAULT_CONFIG
        self.config.update(config)

        # logging boilerplate
        self.logger = logging.getLogger('Mabo')
        formatter = logging.Formatter(log_format)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        self.logger.setLevel(log_level)

        self.shelf_file = 'bobi.shf'

        # query and store
        bways, hways = self._query()
        self.downloaded_buildings = bways.ways
        self.downloaded_highways = hways.ways

    def compute(self, parts, **partitioning_kwargs):

        buildings_cds = self._default_building_unpack()
        highways = self.downloaded_highways
        conn_weigths = CONN_WEIGHTS

        gr = self._graphize_streets(highways)
        self._add_buildings(gr, buildings_cds)
        self._compute_links(gr, conn_weigths, parts, partitioning_kwargs)

    def _query_nkd(self):
        # we load the queries
        ovy = Overpass()
        bquery = self._query_from_file(self.box, os.path.join(this_dir, '../queries/bquery.txt'))
        hquery = self._query_from_file(self.box, os.path.join(this_dir, '../queries/wquery.txt'))

        bways = ovy.query(bquery)
        hways = ovy.query(hquery)

        return bways, hways

    def _query(self):
        # we load the queries
        ovy = _SplitQueryOverpass()
        bquery = self._query_from_file(self.box, os.path.join(this_dir, '../queries/bquery.txt'))
        hquery = self._query_from_file(self.box, os.path.join(this_dir, '../queries/wquery.txt'))

        if self.shelf_file is not None:
            sh = shelve.open(self.shelf_file)

            try:
                # load
                bways_raw, hways_raw, fb, fh = sh[str(hash(tuple(self.box)))]
                if fb.code > 400 or fh.code > 400:  # meaning something went wrong last time
                    raise KeyError
                self.logger.debug('The box was found in the shelf')
            except KeyError:
                # download
                self.logger.debug('Requesting box...')
                bways_raw, fb = ovy.query_raw(bquery)
                hways_raw, fh = ovy.query_raw(hquery)
                self.logger.debug('Box downloaded...')
                # save
                sh[str(hash(tuple(self.box)))] = (bways_raw, hways_raw, fb, fh)
                self.logger.debug('The box was queried and shelved')
            finally:
                sh.close()

            # raw results were loaded, so we have to parse them
            bways = ovy.conclude_raw_query(bways_raw, fb)
            hways = ovy.conclude_raw_query(hways_raw, fh)

        else:
            bways = ovy.query(bquery)
            hways = ovy.query(hquery)

        return bways, hways

    def _default_building_unpack(self):

        bcds = []
        for b in self.downloaded_buildings:
            bcenter = way_center(b)
            barea = way_area(b)
            building_complete_dict = {'id': b.id, 'overpy_way': b, 'center': bcenter, 'area': barea}
            bcds.append(building_complete_dict)
        return bcds

    def _add_buildings(self, gr, from_list):
        if from_list is None:
            bcds = self._default_building_unpack()
        else:
            bcds = from_list

        # add the buildings nodes with attributes
        for building_complete_dict in bcds:
            gr.add_node(building_complete_dict['id'], type='load', building=building_complete_dict,
                        pos=building_complete_dict['center'])

    def _compute_links(self, gr, conn_weigths, parts, partitioning_kwargs):

        total_connect(gr,
                      [b.id for b in self.downloaded_buildings],
                      {b.id: str(b.id) + '_a' for b in self.downloaded_buildings},
                      connection_type_weigths=conn_weigths,
                      phases=1,
                      type='link',
                      logger=self.logger)

        street_nodes = [n for n in gr.nodes if gr.nodes[n]['type'] in ('street', 'street_link')]
        self.logger.debug('Minimum spanning tree calculation...')
        tc = deepcopy(gr)

        # clustering
        self.logger.debug('Cluster elaboration...')
        valid_nodes = [n for n in gr.nodes if gr.nodes[n]['type'] == 'load']

        for n in tc.nodes:
            if tc.nodes[n]['type'] == 'load':
                tc.nodes[n]['pwr'] = tc.nodes[n]['building']['area'] * 0.01
            else:
                tc.nodes[n]['pwr'] = 0.01

        tot_pwr = sum(nx.get_node_attributes(tc, 'pwr').values())

        start_time = time()
        clusters = partition_grid(tc, [x * tot_pwr for x in parts], 'virtual_length', 'pwr', partitioning_kwargs)
        end_time = time()
        self.logger.info(f'Clustering step completed in {end_time - start_time:.2f} seconds')

        loads_in_clusters = [len([n for n in cluster if gr.nodes[n]['type'] == 'load']) for cluster in clusters]
        self.logger.debug('{} clusters created of cardinality: {}'.format(len(clusters), loads_in_clusters))

        # knowing the clusters, now we make n copies of the original graph and we generate the clustered subgraph
        # by clipping away what's unneeded
        subcoms = []
        for clno, cluster in enumerate(clusters):
            self.logger.debug('Subgraph #{} creation and clipping...'.format(clno))
            gr_part = deepcopy(tc)
            relabel = {x: str(x) + '_' + str(clno) for x in street_nodes}
            # each cluster must have its own street nodes, because it's possible that the street paths overlap partially
            nx.relabel_nodes(gr_part, relabel, copy=False)

            # calculating best traf position
            no = deepcopy(gr_part.nodes)
            realnodes = [n for n in no if gr_part.nodes[n]['type'] == 'load' and n in cluster]
            gro_part = nxs.steiner_tree(gr_part, realnodes)

            barycenter = nx.barycenter(gro_part, weight='virtual_length')[0]

            gr_part.add_edge('trmain' + str(clno), barycenter, phases=3, type='entry')
            gr_part.nodes['trmain' + str(clno)]['type'] = 'source'
            gr_part.nodes['trmain' + str(clno)]['pos'] = [x+0.000005 for x in gr_part.nodes[barycenter]['pos']]

            # removing loads not pertaining to this cluster (refines input)
            gr_part.remove_nodes_from([n for n in no if gr_part.nodes[n]['type'] == 'load' and n not in cluster])

            # steiner
            # gr_part = nxs.steiner_tree(gr_part, realnodes, weight='virtual_length')

            # mst
            gr_part = nx.minimum_spanning_tree(gr_part, weight='virtual_length')

            # remove street nodes that are not used by this cluster
            sps = nx.shortest_path(gr_part, 'trmain' + str(clno), weight='virtual_length')
            sps = {k: v for k, v in sps.items() if k in valid_nodes and k in cluster}
            used_nodes = set()
            for target, path in sps.items():
                if gr_part.nodes[target]['type'] in ('street', 'street_link'):
                    raise ValueError
                else:
                    used_nodes.update(set(path))
            unused_nodes = set(gr_part.nodes) - used_nodes
            gr_part.remove_nodes_from(unused_nodes)

            assert nx.number_connected_components(gr_part) <= 1
            assert nx.is_tree(gr_part)

            subcoms.append(gr_part)

        # union of the subgraphs
        try:
            self.g = nx.union_all(subcoms)
        except nx.NetworkXError:
            from itertools import combinations
            ls = [(n, set(x.nodes)) for n, x in enumerate(subcoms)]
            for n1x, n2y in combinations(ls, 2):
                n1, x = n1x
                n2, y = n2y
                print('{},{} : {}'.format(n1, n2, x.intersection(y)))
            raise

        self.logger.info('Grid created')

    @staticmethod
    def _query_from_file(box, file):
        with open(file, 'r') as bq:
            qry = bq.read()

        sbox = [str(x) for x in box]
        qry = qry.format(','.join(sbox))
        return qry

    def _graphize_streets(self, highways, node_suffix=''):

        self.logger.debug('Creating street skeleton graph...')
        street_graph = nx.Graph()

        for h in highways:
            raw_ns = h.get_nodes()

            for n1, n2 in _walk2(raw_ns):

                # trimming streets out of box
                if not ((self.box[0] < n1.lat < self.box[2]) and (self.box[1] < n1.lon < self.box[3])):
                    if not ((self.box[0] < n2.lat < self.box[2]) and (self.box[1] < n2.lon < self.box[3])):
                        continue

                street_graph.add_node(str(n2.id) + node_suffix,
                                      pos=asarray([n1.lon, n1.lat]).astype(float),
                                      type='street')
                street_graph.add_node(str(n1.id) + node_suffix,
                                      pos=asarray([n2.lon, n2.lat]).astype(float),
                                      type='street')
                street_graph.add_edge(str(n2.id) + node_suffix, str(n1.id) + node_suffix, phases=3, type='street')

        # it may be possible that the streets are not connected. In that case, we only take the biggest component.
        # one could also choose to add nonexistent street edges or to find the minimum street path to connect
        # everything with openstreetmaps (dangerous).
        if not nx.is_connected(street_graph):
            self.logger.warning('The original graph is not connected. Only the biggest component will be kept.')
            subgraphs = [nx.subgraph(street_graph, n) for n in nx.connected_components(street_graph)]
            sizes = {len(x.nodes): x for x in subgraphs}
            biggest_sg = sizes[max(sizes.keys())]

            killnodes = [n for n in street_graph.nodes if n not in biggest_sg.nodes]
            street_graph.remove_nodes_from(killnodes)

        return street_graph

    def subplot(self, go=DEFAULT_GRAPHIC_OPTS, **nx_opts):

        clr = get_cmap('tab10')

        nl_tot = set([b.id for b in self.downloaded_buildings])
        ccs = [nx.subgraph(self.g, n) for n in nx.connected_components(self.g)]

        for idx, sgee in enumerate(ccs):
            eggos = list(sgee.edges)
            color_component = clr(idx / 10 + 0.02)

            nl_this = list(nl_tot.intersection(set(sgee.nodes)))

            nxo = dict(pos=nx.get_node_attributes(sgee, 'pos'),
                       node_size=go['node_size'],
                       font_size=go['font_size'],
                       edgelist=eggos,
                       nodelist=[],
                       edge_color=[color_component] * len(sgee.edges),
                       cmap=colors.Colormap('hot'))
            nxo.update(nx_opts)

            nx.draw_networkx(sgee, **nxo, with_labels=False)

            # bigger building nodes
            nx.draw_networkx_nodes(sgee, nx.get_node_attributes(sgee, 'pos'), nl_this, go['bnode_size'],
                                   node_color=[color_component] * len(nl_this), **nx_opts)

            # white center
            nx.draw_networkx_nodes(sgee, nx.get_node_attributes(sgee, 'pos'), nl_this, go['bnode_size']*0.5,
                                   node_color='#ffffff', **nx_opts)

        _paint_map(self.box, plt, go, self.logger)
        plt.axis('equal')
        plt.show()




