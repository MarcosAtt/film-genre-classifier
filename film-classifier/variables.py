import numpy as np

genre_name = [
    "crime",
    "romance",
    "science fiction",
    "western",
]

default_Q = 1000
default_seed = 42


def print_avance(i, j, iMax, jMax):
    total = (iMax) * (jMax)
    porcentaje = (i * jMax) + (j)
    if i * j < total:
        print(int(porcentaje / total * 100), "%", end="\r", flush=True)
    if (i + 1) * (j + 1) == total:
        print("100 %!")


def matrizHouseholder(D, v):
    """Devuelve matriz Householder I (forma de D) - 2 v vt"""
    "v vector unitario"
    return np.eye(D.shape[0]) - 2 * (v @ v.T)


def trucoHouseholder(D):
    """D debe ser una matriz diagonal"""
    v = np.random.rand(D.shape[0], 1)
    v = v / np.linalg.norm(v)
    H = matrizHouseholder(D, v)
    return H @ D @ H.T
