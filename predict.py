import sys
import numpy as np
from network_utils import generate_G_matrix, generate_yhad_matrix, get_precision, evaluate_parameters
from main import prepare_data
from plotter import plot_two_class_classification, plot_regression_data

if __name__ == '__main__':
    path_to_test_data = str(sys.argv[1])
    TYPE_OF_PROBLEM = int(sys.argv[4])

    W = np.load("weights.npy")
    V = np.load("V.npy")
    gama = np.load("gama.npy")

    X, y, L, n, scales = prepare_data(number_of_lines=float("inf"), path=path_to_test_data)
    m, n_prime = V.shape
    assert n == n_prime
    G = generate_G_matrix(X, V, gama)
    yhad = generate_yhad_matrix(G, W)

    if TYPE_OF_PROBLEM == 0:
        plot_regression_data(yhad)
    elif TYPE_OF_PROBLEM == 1:
        print("Error on test data set:", evaluate_parameters(V, gama, X, y))
        print("Precision on test data set:", get_precision(X, y, V, gama))
        yhad = np.where(yhad > 0, 1, -1)
        plot_two_class_classification(X, y, V, yhad, scales)
    elif TYPE_OF_PROBLEM == 2:
        # TODO
        pass
