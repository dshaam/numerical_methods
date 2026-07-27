import numpy as np


def spline_S10(x, n, x_nodes, y_nodes):
    S = np.zeros_like(x)

    for j, cur_x in enumerate(x):
        for i in range(n - 1):
            if x_nodes[i] <= cur_x <= x_nodes[i + 1]:
                x1, x2 = x_nodes[i], x_nodes[i + 1]
                y1, y2 = y_nodes[i], y_nodes[i + 1]

                S[j] = y1 + ((y2 - y1) * (cur_x - x1)) / (x2 - x1)

                break
    return S


def spline_S21(x, n, x_nodes, y_nodes):
    a = np.array(y_nodes)
    b = np.zeros(n)
    c = np.zeros(n)
    b[0] = 0  # хотя если поставить так (a[1] - a[0]) / (x_nodes[1] - x_nodes[0]),
    # то приближение как будто лучше, но тогда первый интервал является прямой

    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        c[i] = (a[i + 1] - a[i] - b[i] * h) / (h * h)
        b[i + 1] = b[i] + 2 * c[i] * h
    S = np.zeros_like(x)
    #print(a, b, c)

    for j, cur_x in enumerate(x):
        for i in range(n - 1):
            if x_nodes[i] <= cur_x <= x_nodes[i + 1]:
                S[j] = a[i] + b[i] * (cur_x - x_nodes[i]) + c[i] * (cur_x - x_nodes[i]) * (cur_x - x_nodes[i])
                break
    return S

def spline_S32(x, n, x_nodes, y_nodes):
    a = np.array(y_nodes)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    h = np.zeros(n - 1)
    for i in range(n - 1):
        h[i] = x_nodes[i + 1] - x_nodes[i]

    H = np.zeros((n, n))
    gamma = np.zeros(n)

    H[0, 0] = 1
    H[n - 1, n - 1] = 1

    for i in range(1, n - 1):
        H[i, i - 1] = h[i - 1]
        H[i, i] = 2 * (h[i - 1] + h[i])
        H[i, i + 1] = h[i]
        gamma[i] = 3 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])

    c = np.linalg.solve(H, gamma)

    for i in range(n - 1):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    S = np.zeros_like(x)

    for j, cur_x in enumerate(x):
        for i in range(n - 1):
            if x_nodes[i] <= cur_x <= x_nodes[i + 1]:
                dx = cur_x - x_nodes[i]
                S[j] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
                break

    return S