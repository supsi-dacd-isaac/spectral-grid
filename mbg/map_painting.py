from functools import lru_cache
from math import radians as rad

import staticmap as sm
from numpy import mean, cos

DEFAULT_GRAPHIC_OPTS = {
    'font_size': 9,
    'px_mult': 9e5,
    'node_size': 100,
    'bnode_size': 100,
    'zoom': 18
}


def _resolution_x(zoom):
    return 360 / 256.0 / (2 ** zoom)


def _resolution_y(zoom, latitude):
    return 360 / 256.0 * cos(rad(latitude)) / (2 ** zoom)


def _center(box):
    return mean([box[1], box[3]]), mean([box[0], box[2]])


@lru_cache()
def _request_map(center, zoom, px_bounds):
    pppx = _resolution_x(zoom)
    pppy = _resolution_y(zoom, center[1])

    i = sm.StaticMap(*px_bounds)
    img = i.render(zoom=zoom, center=center)

    box = [center[0] - px_bounds[0] / 2 * pppx,  # minx
           center[0] + px_bounds[0] / 2 * pppx,  # maxx
           center[1] - px_bounds[1] / 2 * pppy,  # miny
           center[1] + px_bounds[1] / 2 * pppy,  # maxy
           ]

    return img, box


def _paint_map(box, plt_proxy, graphical_options=DEFAULT_GRAPHIC_OPTS, logger=None):
    """paints the map inside the rectangle described by box onto the object plt_proxy.
    plt_proxy has to have an attribute 'imshow' that behaves like matplotlib.pyplot.imshow ."""

    def _req_map(h, w, center, zoom):
        if logger is not None:
            logger.info('requesting map...')
        img, im_box = _request_map(center=center, zoom=zoom, px_bounds=(w, h))
        if logger is not None:
            logger.info('map downloaded')
        return img, im_box

    h = int((box[2] - box[0]) * graphical_options['px_mult'])
    w = int((box[3] - box[1]) * graphical_options['px_mult'])
    center = _center(box)
    zoom = graphical_options['zoom']

    img, im_box = _req_map(h, w, center, zoom)

    if logger is not None:
        logger.debug('Plotting...')
    plt_proxy.imshow(img, zorder=0, extent=im_box, cmap='gray', interpolation='none')