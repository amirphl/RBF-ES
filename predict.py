import sys
import numpy as np
from network_utils import generate_G_matrix, generate_yhad_matrix, get_precision, evaluate_parameters
from main import prepare_data, prepare_data_for_multi_class_classification
from plotter import plot_two_class_classification, plot_regression_data, plot_multi_class_classification

if __name__ == '__main__':
    path_to_test_data = str(sys.argv[1])
    nol = int(sys.argv[2])  # number of lines of test data
    type_of_problem = sys.argv[3]
    radius_scale = float(sys.argv[4])
    sys.argv[4] = sys.argv[3]

    W = np.load("outputs/weights" + type_of_problem + ".npy")
    V = np.load("outputs/V" + type_of_problem + ".npy")
    gama = np.load("outputs/gama" + type_of_problem + ".npy")

    type_of_problem = int(type_of_problem)

    X, y, scales = prepare_data(number_of_lines=nol, path=path_to_test_data)
    l, n = X.shape
    c = 0
    if type_of_problem == 2:
        y = prepare_data_for_multi_class_classification(y)
        _, c = y.shape

    m, n_prime = V.shape
    assert n == n_prime

    G = generate_G_matrix(X, V, gama)
    yhad = generate_yhad_matrix(G, W)

    if type_of_problem == 0:
        plot_regression_data(yhad)
    elif type_of_problem == 1:
        print("Error on test data set:", evaluate_parameters(V, gama, X, y))
        print("Precision on test data set:", get_precision(X, y, V, gama))
        yhad = np.where(yhad > 0, 1, -1)
        plot_two_class_classification(X, y, V, gama, yhad, radius_scale)
    elif type_of_problem == 2:
        print("Error on test data set:", evaluate_parameters(V, gama, X, y))
        print("Precision on test data set:", get_precision(X, y, V, gama))
        b = np.zeros_like(yhad)
        b[np.arange(len(yhad)), yhad.argmax(1)] = 1
        yhad = b
        plot_multi_class_classification(X, y, V, gama, yhad, radius_scale)
