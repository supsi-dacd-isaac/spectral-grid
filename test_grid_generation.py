import matplotlib

from spectral_grid import MapBoxGraph, DEFAULT_CONN_WEIGHTS


if __name__ == '__main__':

    matplotlib.use('Qt5Agg')

    AROSIO = (46.043510148612334,
              8.894011974334717,
              46.05036097561633,
              8.908238410949707)

    GRAPHIC_OPTS = {
        'node_shape': 'h',
    }

    mbg = MapBoxGraph(AROSIO, None, log_level=10)
    mbg.compute([.25, .35, .4], DEFAULT_CONN_WEIGHTS, maxiter=100, imbalance_tol=1e-1)
    mbg.subplot(**GRAPHIC_OPTS)
