import numpy as np
import pandas as pd

from lagrange import polinom_lagrange
from newton import polinom_newton
from functions import f, opt_nodes_formula
from splines import spline_S10, spline_S21, spline_S32
from visual import visualize, culculation_error_graph


def main():
    a = -0.5
    b = 2

    number_of_nodes = [3, 4, 5, 7, 10, 30, 40, 50, 80, 100, 150, 200]

    interpolation_table = []
    spline_table = []

    for n in number_of_nodes:
        m = max(min(n * n, 4000), 100)
        x = np.linspace(a, b, m)
        x_equidistant_nodes = np.linspace(a, b, n)
        x_opt_nodes = opt_nodes_formula(a, b, n)

        y = f(x)
        y_equidistant_nodes = f(x_equidistant_nodes)
        y_opt_nodes = f(x_opt_nodes)

        def L_n(x):
            return polinom_lagrange(x, n, x_equidistant_nodes)

        def Lopt_n(x):
            return polinom_lagrange(x, n, x_opt_nodes)

        Rln = np.max(np.abs(f(x) - L_n(x)))
        Rlopt = np.max(np.abs(f(x) - Lopt_n(x)))

        visualize(
            x,
            [x_equidistant_nodes, x_opt_nodes],
            [y_equidistant_nodes, y_opt_nodes],
            f, 2,
            [L_n, Lopt_n],
            ["Полином Лагранжа по равноотстоящим узлам", "Полином Лагранжа по оптимальным узлам"],
            ["Равноудаленные узлы", "Оптимальные узлы"],
            f"Lagrange_{n}_nodes.jpeg"
        )

        def N_n(x):
            return polinom_newton(x, x_equidistant_nodes, n)

        def Nopt_n(x):
            return polinom_newton(x, x_opt_nodes, n)

        Rnn = np.max(np.abs(y - N_n(x)))
        Rnoptn = np.max(np.abs(y - Nopt_n(x)))

        interpolation_table.append({
            "Количество узлов (n)": n,
            "Количество проверочных точек (m)": m,
            "Максимальное отклонение (RLn)": Rln,
            "Максимальное отклонение (RLoptn)": Rlopt,
            "Максимальное отклонение (RNn)": Rnn,
            "Максимальное отклонение (RNoptn)": Rnoptn
        })

        visualize(
            x,
            [x_equidistant_nodes, x_opt_nodes],
            [y_equidistant_nodes, y_opt_nodes],
            f, 2,
            [N_n, Nopt_n],
            ["Полином Ньютона по равноотстоящим узлам", "Полином Ньютона по оптимальным узлам"],
            ["Равноудаленные узлы", "Оптимальные узлы"],
            f"Newton_{n}_nodes.jpeg"
        )

        def S10(x):
            return spline_S10(x, n, x_equidistant_nodes, y_equidistant_nodes)

        def S21(x):
            return spline_S21(x, n, x_equidistant_nodes, y_equidistant_nodes)

        def S32(x):
            return spline_S32(x, n, x_equidistant_nodes, y_equidistant_nodes)

        visualize(
            x,
            [x_equidistant_nodes, x_equidistant_nodes, x_equidistant_nodes],
            [y_equidistant_nodes, y_equidistant_nodes, y_equidistant_nodes],
            f, 3,
            [S10, S21, S32],
            ["Линейный сплайн", "Квадратичный сплайн", "Кубический сплайн"],
            ["Равноудаленные узлы", "Равноудаленные узлы", "Равноудаленные узлы"],
            f"Spline_{n}_nodes.jpeg"
        )

        RS10 = np.max(np.abs(y - S10(x)))
        RS21 = np.max(np.abs(y - S21(x)))
        RS32 = np.max(np.abs(y - S32(x)))

        spline_table.append({
            "Количество узлов (n)": n,
            "Количество проверочных точек (k)": m,
            "RS10": RS10,
            "RS21": RS21,
            "RS32": RS32
        })

        err_lagr = np.abs(y - L_n(x))
        err_spline = np.abs(y - S32(x))

        culculation_error_graph(x, n, err_lagr, err_spline, f"error_plot_{n}_nodes.jpeg")

    df_interpolation = pd.DataFrame(interpolation_table)
    df_spline = pd.DataFrame(spline_table)
    df_interpolation.to_csv("interpolation_table.csv", index=False)
    df_spline.to_csv("spline_table", index=False)

    '''print("Таблица для интерполяции:")
    print(df_interpolation.round(6).to_string(index=False))

    print("\nТаблица для сплайнов:")
    print(df_spline.round(6).to_string(index=False))'''


main()