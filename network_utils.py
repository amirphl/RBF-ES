import numpy as np
import sys


def g_function(x, v, gama):
    temp = np.subtract(x, v).T
    return np.exp(-gama * np.inner(temp, temp))


def generate_G_matrix(X, V, gama):
    (l, n) = X.shape
    (m, _) = V.shape
    G = np.ndarray(shape=(l, m), dtype=np.float32)
    i = 0

    for x in X:
        j = 0
        for v, g in zip(V, gama[0, :]):
            G[i, j] = g_function(x, v, g)
            j = j + 1
        i = i + 1
    return G


def generate_W_matrix(G, y):
    try:
        res = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), y)
        return res
    except np.linalg.LinAlgError:
        print("..")
        _, m = G.shape
        type_of_problem = int(sys.argv[4])
        # TODO is it OK to fill with random numbers?
        res = None
        if type_of_problem == 0 or type_of_problem == 1:
            res = np.random.uniform(low=-1, high=1, size=m)
        elif type_of_problem == 2:
            _, c = y.shape
            res = np.random.uniform(low=-1, high=1, size=m * c).reshape(m, c)
        assert res is not None
        return res


def generate_yhad_matrix(G, W):
    return np.dot(G, W)


def mse(y, yhad):
    diff = np.subtract(y, yhad)
    return 0.5 * np.inner(diff, diff)


def evaluate_parameters(V, gama, X, y):
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)
    yhad = generate_yhad_matrix(G, W)
    error = None
    type_of_problem = int(sys.argv[4])

    if type_of_problem == 0:
        error = mse(y, yhad)
    elif type_of_problem == 1:
        yhad = np.where(yhad > 0, 1, -1)
        error = mse(y, yhad) / 2
    elif type_of_problem == 2:
        # TODO bad idea
        l, _ = yhad.shape
        error = 0
        b = np.zeros_like(yhad)
        b[np.arange(len(yhad)), yhad.argmax(1)] = 1
        yhad = b
        abs_diff = np.abs(np.subtract(yhad, y))
        error = np.sum(abs_diff) / 2
    assert error is not None
    return error


def get_precision(X, y, V, gama):
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)
    yhad = generate_yhad_matrix(G, W)
    L, _ = G.shape
    precision = None
    type_of_problem = int(sys.argv[4])

    if type_of_problem == 1:
        yhad = np.where(yhad > 0, 1, -1)
        precision = 1 - mse(y, yhad) / (2 * L)
    elif type_of_problem == 2:
        b = np.zeros_like(yhad)
        b[np.arange(len(yhad)), yhad.argmax(1)] = 1
        yhad = b
        abs_diff = np.abs(np.subtract(yhad, y))
        precision = 1 - np.matrix.sum(abs_diff) / (2 * L)
    assert precision is not None
    return precision
