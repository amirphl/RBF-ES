import matplotlib.pyplot as plt
import numpy as np


def plot_two_class_classification(X, y, V, yhad):
    L, n = X.shape
    assert n == 2

    mapping = {-1: ("red", "."), 1: ("blue", "."), -2: ("orange", "."), 2: ("black", ".")}

    for i in range(L):
        if yhad[i] != y[i]:
            if yhad[i] == 1:
                yhad[i] = -2
            else:
                yhad[i] = 2

    for c in np.unique(yhad):
        d = X[yhad == c]
        plt.scatter(d[:, 0], d[:, 1], c=mapping[c][0])
    plt.scatter(V[:, 0], V[:, 1], c="green")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.suptitle('2 class classification')
    plt.savefig('2_class_classification_output.png')
    plt.show()



def plot_regression_data(yhad):
    L = yhad.shape
    plt.scatter(range(L[0]), yhad)
    plt.xlabel('position of data')
    plt.ylabel('regression bound')
    plt.suptitle('regression output')
    plt.savefig('regression output.png')
    plt.show()
