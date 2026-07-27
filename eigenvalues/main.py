import numpy as np
from power_method import power_method
from inverse_iteration import inverse_iteration
from QR_algorithm import qr_algorithm


def main():
    n = 6
    diag_elements = np.random.randint(-10, 10, size=n)
    lambdaa = np.diag(diag_elements)
    C = np.random.randint(-10, 10, size=(n, n))
    C_inverse = np.linalg.inv(C)
    A = C_inverse @ lambdaa @ C

    print(lambdaa)
    # print(A.round(3))
    # print(*power_method(A))
    results = [inverse_iteration(A, mu) for mu in (diag_elements - 0.2)]

    for _, (lam, vec) in enumerate(results, 1):
        print(f"lambda = {lam}")
        print("v =", np.round(vec, 4))
        print()
    #print(qr_algorithm(A)[0])

main()