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
        # print("SVD Convergence Error.")
        print("..")
        _, m = G.shape
        TYPE_OF_PROBLEM = int(sys.argv[4])
        # TODO is it OK to fill with random numbers?
        res = None
        if TYPE_OF_PROBLEM == 0 or TYPE_OF_PROBLEM == 1:
            res = np.random.uniform(low=-1, high=1, size=m)
        elif TYPE_OF_PROBLEM == 2:
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
    TYPE_OF_PROBLEM = int(sys.argv[4])
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)
    yhad = generate_yhad_matrix(G, W)
    error = None
    if TYPE_OF_PROBLEM == 0:
        error = mse(y, yhad)
    elif TYPE_OF_PROBLEM == 1:
        yhad = np.where(yhad > 0, 1, -1)
        error = mse(y, yhad) / 2
    elif TYPE_OF_PROBLEM == 2:
        # TODO
        pass
    return error


def get_precision(X, y, V, gama):
    TYPE_OF_PROBLEM = int(sys.argv[4])
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)
    yhad = generate_yhad_matrix(G, W)
    L, _ = G.shape
    precision = None
    if TYPE_OF_PROBLEM == 1:
        yhad = np.where(yhad > 0, 1, -1)
        precision = 1 - mse(y, yhad) / (2 * L)
    elif TYPE_OF_PROBLEM == 2:
        # TODO
        pass
    return precision


def get_train_error(X, y, V, gama):
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)
    yhad = generate_yhad_matrix(G, W)
    error = mse(y, yhad)
    return error
