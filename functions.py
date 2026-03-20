import numpy as np


def f(x):
    return x * np.log(x + 1)


def opt_nodes_formula(a, b, n):
    x_opt = ((b - a) * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * (n + 1))) + (b + a)) / 2
    # print(*x, sep='\n')
    return x_opt