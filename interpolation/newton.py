import functions
import numpy as np


def divided_difference(x, n):
    table = np.zeros((n, n))
    table[:, 0] = functions.f(x)

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    # print(table)
    return table


def polinom_newton(x, x_nodes, n):
    coef = divided_difference(x_nodes, n)[0]
    ans = coef[0]
    product = 1

    for i in range(1, n):
        product *= (x - x_nodes[i - 1])
        ans += coef[i] * product
    return ans