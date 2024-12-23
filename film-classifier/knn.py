import eigenvalues
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle  # Para guardar variables
from numpy import linalg as LA
from scipy import stats


def distancia_coseno(dev, train):
    # distancias[i][j] = dist(dev[i], train[j])
    return np.ones((len(dev), len(train))) - dev @ train.T


def distancia_euclidea(A, B):
    dist = np.zeros((len(A), len(B)), dtype=float)
    for i in range(len(A)):
        for j in range(len(B)):
            dist[i, j] = np.linalg.norm(A[i, :] - B[j, :])

    return dist


def calcular_vecinos(X_train, X_dev, dist_cos=True):
    distancias = 0
    if dist_cos:
        distancias = distancia_coseno(X_dev, X_train)
    else:
        distancias = distancia_euclidea(X_dev, X_train)
    vecinos = np.argsort(distancias, axis=1)  # Ordenar por distancia
    return vecinos


def clasificar(vecinos, y_train, k):
    # En cada elemento devuelve la moda correspondiente
    modas = stats.mode(y_train[vecinos[::, :k]], axis=1)[0]
    return modas


def contar_predicciones_correctas(guesses, y_dev):
    correctas = len(guesses) - np.count_nonzero(guesses - y_dev)
    return correctas


def medir_exactitud(vecinos, y_train, y_dev, k) -> float:
    predicciones = clasificar(vecinos, y_train, k)
    correctas = contar_predicciones_correctas(predicciones, y_dev)
    totales = len(y_dev)
    return correctas / totales


def normalize_data(train_data):
    train_normas = np.diag((1 + 1) / (np.linalg.norm(train_data, axis=1) + 1))
    train_normalized = train_normas @ train_data
    return train_normalized
