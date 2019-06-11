import numpy as np


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
        print("SVD Convergence Error.")
        _, m = G.shape
        # TODO is it OK to fill with random numbers?
        res = np.random.uniform(low=-1, high=1, size=m)
        return res


def generate_yhad_matrix(G, W):
    return np.dot(G, W)


def mse(y, yhad):
    diff = np.subtract(y, yhad)
    return 0.5 * np.inner(diff, diff)


def evaluate_parameters(V, gama, X, y):
    from main import TYPE_OF_PROBLEM
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
    from main import TYPE_OF_PROBLEM
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
