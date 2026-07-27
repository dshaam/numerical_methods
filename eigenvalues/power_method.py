import numpy as np


def power_method(A, max_iter=10000):
    n = A.shape[0]
    delta = 1e-8  # по условию
    rtol = 1e-6  # по условию
    y = np.ones(n)  # по конспекту y0
    z = y / np.linalg.norm(y)  # по конспекту z0
    labmda = A[0][0]

    for _ in range(max_iter):
        new_y = A @ z
        new_z = new_y / np.linalg.norm(new_y)
        available_z = np.abs(z) > delta
        cur_labmda = np.mean(new_y[available_z] / z[available_z])
        if abs(labmda - cur_labmda) / abs(cur_labmda) < rtol:
            break
        labmda = cur_labmda
        z = new_z
    return cur_labmda, z
