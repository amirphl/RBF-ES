import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_two_class_classification(X, y, V, gama, yhad, radius_scale):
    L, n = X.shape
    assert n == 2

    mapping = {-1: ("red", "."), 1: ("blue", "."), -2: ("orange", "."), 2: ("black", ".")}

    plt.figure()

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

    ax = plt.gca()
    i = 0
    print(gama[0, :])
    for radius in gama[0, :]:
        circle = plt.Circle((V[i, 0], V[i, 1]), radius=radius / radius_scale, color='red', fill=False)
        ax.add_artist(circle)
        i += 1

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.suptitle('2 class classification on test file: ' + sys.argv[1])
    plt.savefig('outputs/2_class_classification_output.png')
    plt.show()
    exit(0)


def plot_multi_class_classification(X, y, V, gama, yhad, radius_scale):
    L, n = X.shape
    assert n == 2

    mapping = {-1: ("red", "."), 0: ("purple", "."), 1: ("blue", "."), 2: ("green", "."), 3: ("black", "."),
               4: ("pink", "."), 5: ("yellow", "."), 6: ("orange", ".")}
    y = np.argmax(y, axis=1)
    for i in range(L):
        if yhad[i, int(y[i])] == 1:
            y[i] = int(y[i])
        else:
            y[i] = -1

    plt.figure()

    temp = y
    for c in np.unique(y):
        d = X[temp == c]
        plt.scatter(d[:, 0], d[:, 1], c=mapping[c][0])

    plt.scatter(V[:, 0], V[:, 1], c="green")

    ax = plt.gca()
    i = 0
    print(gama[0, :])
    for radius in gama[0, :]:
        circle = plt.Circle((V[i, 0], V[i, 1]), radius=radius / radius_scale, color='red', fill=False)
        ax.add_artist(circle)
        i += 1

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.suptitle('multi class classification on test file: ' + sys.argv[1])
    plt.savefig('outputs/multi_class_classification_output.png')
    plt.show()


def plot_regression_data(yhad):
    L = yhad.shape
    plt.scatter(range(L[0]), yhad)
    plt.xlabel('')
    plt.ylabel('regression range')
    plt.suptitle('regression output on test file: ' + sys.argv[1])
    plt.savefig('outputs/regression output.png')
    plt.show()
