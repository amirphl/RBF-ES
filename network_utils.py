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
        for v, g in zip(V, gama):
            G[i, j] = g_function(x, v, g)
            j = j + 1
        i = i + 1
    return G


def generate_W_matrix(G, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), y)


def generate_yhad_matrix(G, W):
    return np.dot(G, W)


def mse(y, yhad):
    diff = np.subtract(y, yhad)
    return 0.5 * np.inner(diff, diff)


def evaluate_parameters(V, gama, X, y):
    G = generate_G_matrix(X, V, gama)
    W = generate_W_matrix(G, y)
    yhad = generate_yhad_matrix(G, W)
    error = mse(y, yhad)
    return error
