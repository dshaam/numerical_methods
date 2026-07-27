import numpy as np


def inverse_iteration(A, mu=0.0, max_iter=10000):
    delta = 1e-8  # по условию
    rtol = 1e-6  # по условию

    n = A.shape[0]
    y = np.ones(n)
    z = y / np.linalg.norm(y)
    E = np.eye(n)
    labmda = 0

    for _ in range(max_iter):
        new_y = np.linalg.solve(A - mu * E, z)
        new_z = new_y / np.linalg.norm(new_y)
        available_y = np.abs(new_y) > delta
        cur_labmda = np.mean(z[available_y] / new_y[available_y])
        if abs(labmda - cur_labmda) / abs(cur_labmda) < rtol:
            break
        labmda = cur_labmda
        z = new_z

    return cur_labmda + mu, new_z
