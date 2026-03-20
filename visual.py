import matplotlib.pyplot as plt
from os import makedirs, path


def visualize(x, x_nodes, y_nodes, original_function, number_of_functions, functions, function_labels, nodes_labels, filename="photo.jpeg"):
    '''
    :param x: точки, которые мы подставляем
    :param x_nodes: значения узлов по х
    :param y_nodes: значения узлов по у
    :param original_function: функция f(x) = x ln(x+1)
    :param number_of_functions: количество графиков
    :param functions: сами графики
    :param function_labels: названия графиков
    :param nodes_labels: название узлов
    :param filename: название файла
    :return:
    '''

    fig, axes = plt.subplots(1, number_of_functions, figsize=(20, 8))

    folder = "graphs"
    makedirs(folder, exist_ok=True)

    filename = path.join(folder, filename)

    if number_of_functions == 1:
        axes = [axes]

    for i in range(number_of_functions):
        axes[i].plot(x, original_function(x), label="f(x) = x ln(x+1)")
        axes[i].plot(x, functions[i](x), label=function_labels[i])
        axes[i].scatter(x_nodes[i], y_nodes[i], color='red', label=nodes_labels[i])
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        #axes[0].set_ylim(-1, 3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def culculation_error_graph(x_test, n, err_lagr, err_spline, filename="error_plot.jpeg"):
    plt.figure(figsize=(10, 6))

    filename = f"graphs/{filename}"

    plt.plot(x_test, err_lagr, label='|f(x) - L_n(x)|')
    plt.plot(x_test, err_spline, label='|f(x) - S_3,2(x)|')
    plt.xlabel('x')
    plt.ylabel('Абсолютная погрешность')
    plt.title(f'Распределение абсолютной погрешности, n={n}')
    plt.grid()
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()