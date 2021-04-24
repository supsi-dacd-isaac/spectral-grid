_zero2D = [0, 0]


def project(P, A, B):
    v = [B[0] - A[0], B[1] - A[1]]
    u = [A[0] - P[0], A[1] - P[1]]
    vu = v[0] * u[0] + v[1] * u[1]
    vv = v[0] ** 2 + v[1] ** 2
    t = -vu / vv
    if 0 < t < 1:
        return _vectorToSegment2D(t, _zero2D, A, B)
    g0 = _sqDiag2D(_vectorToSegment2D(0, P, A, B))
    g1 = _sqDiag2D(_vectorToSegment2D(1, P, A, B))
    if g0 <= g1:
        return A
    else:
        return B


def _vectorToSegment2D(t, P, A, B):
    return [
        (1 - t) * A[0] + t * B[0] - P[0],
        (1 - t) * A[1] + t * B[1] - P[1],
    ]


def _sqDiag2D(P):
    return P[0] ** 2 + P[1] ** 2