import numpy as np
from lagrange import polinom_lagrange
from newton import polinom_newton
from functions import f, opt_nodes_formula
from splines import spline_S10, spline_S21, spline_S32
from visual import visualize, culculation_error_graph


def print_header():
    print(f"{'Количество узлов (n)':<25}"
          f"{'Количество проверочных точек (m)':<35}"
          f"{'Максимальное отклонение (RLn)':<50}"
          f"{'Максимальное отклонение (RLoptn)':<40}"
          f"{'Максимальное отклонение (RNn)':<80}"
          f"{'Максимальное отклонение (RNoptn)':<80}")
    print("-"*300)


def print_table(n, m, rln, rlopt, rnn, rnoptn):
    print(f"{n:<25}{m:<35}{rln:<50.6f}{rlopt:<40.6f}{rnn:<80.6f}{rnoptn:<80.6f}")


def print_header_spline():
    print(f"{'Количество узлов (n)':<25}"
          f"{'Количество проверочных точек (k)':<35}"
          f"{'RS10':<20}"
          f"{'RS21':<20}"
          f"{'RS32':<20}")
    print("-" * 120)


def print_table_spline(n, k, rs10, rs21, rs32):
    print(f"{n:<25}{k:<35}{rs10:<20.6f}{rs21:<20.6f}{rs32:<20.6f}")


def main():
    a = -0.5
    b = 2

    number_of_nodes = [3, 4, 5, 7, 10, 30, 40, 50, 80, 100, 150, 200]

    print_header()
    # print_header_spline()

    for n in number_of_nodes:
        m = max(min(n * n, 4000), 100)
        x = np.linspace(a, b, m)  # ЭТИ ТОЧКИ ПРОСТО ПОДСТАВЛЯЕМ, ПО НИМ НИКАКИХ ПОЛИНОМОВ НЕ СТРОИТЬ
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

        print_table(n, m, Rln, Rlopt, Rnn, Rnoptn)

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

        # print_table_spline(n, m, RS10, RS21, RS32)

        err_lagr = np.abs(y - L_n(x))
        err_spline = np.abs(y - S32(x))

        culculation_error_graph(x, n, err_lagr, err_spline, f"error_plot_{n}_nodes.jpeg")


main()
