import functions
import numpy as np


def polinom_lagrange(x, n, x_nodes):
    ans = 0
    y_nodes = functions.f(x_nodes)
    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])  # Li станет массивом после первой итерации, потому что x это изначально нампаевский массив
        # print(Li)
        ans += y_nodes[i] * Li
    return ans