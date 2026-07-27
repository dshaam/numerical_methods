import numpy as np


def sgn_plus(x):
    return 1.0 if x >= 0 else -1.0


def hessenberg_form(A):
    n = A.shape[0]
    Ak = A.copy()

    for k in range(n - 2):
        x = Ak[k + 1:, k]
        if np.allclose(x[1:], 0):
            continue

        a = Ak[k + 1, k]
        s = sgn_plus(a) * np.linalg.norm(x)

        if np.isclose(s, a):
            continue

        mu = 1.0 / np.sqrt(2.0 * s * (s - a))

        v = np.zeros(n)
        v[k + 1] = mu * (a - s)
        v[k + 2:] = mu * Ak[k + 2:, k]

        Hk = np.eye(n) - 2.0 * np.outer(v, v)  # здесь вмето транспонирования (v1v1 v1v2 v1v2 | v2v1...)

        Ak = Hk @ Ak @ Hk

    return Ak


def givens_rotation(x, y):
    if abs(y) < 1e-15:
        return 1.0, 0.0

    r = np.hypot(x, y)
    c = x / r
    s = -(y / r)
    return c, s


def qr_givens(H):
    n = H.shape[0]
    R = H.copy()
    Q = np.eye(n)

    for k in range(n - 1):
        x = R[k, k]
        y = R[k + 1, k]
        c, s = givens_rotation(x, y)

        G = np.array([[c, -s],
                      [s, c]])

        R[k:k + 2, k:] = G @ R[k:k + 2, k:]
        Q[:, k:k + 2] = Q[:, k:k + 2] @ G.T

    return Q, R


def qr_algorithm(A, max_iter=10000):
    n = A.shape[0]
    H = hessenberg_form(A)
    eigenvalues = np.zeros(n)
    m = n
    epsilon = 1e-8

    total_iter = 0

    while m > 1:
        if abs(H[m - 1, m - 2]) < epsilon:
            H[m - 1, m - 2] = 0.0
            eigenvalues[m - 1] = H[m - 1, m - 1]
            m -= 1
            continue

        if total_iter > max_iter:
            return None

        mu = H[m - 1, m - 1]

        Hm = H[:m, :m] - mu * np.eye(m)
        Q, R = qr_givens(Hm)
        H[:m, :m] = R @ Q + mu * np.eye(m)

        total_iter += 1

    eigenvalues[0] = H[0, 0]
    return eigenvalues, H