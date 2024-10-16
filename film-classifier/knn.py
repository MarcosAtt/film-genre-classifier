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


def calcular_vecinos(X_train, X_dev):
    distancias = distancia_coseno(X_dev, X_train)  # TODO: generalizar
    vecinos = np.argsort(distancias, axis=1)  # Ordenar por distancia
    return vecinos


def moda_cada_elemento(vecinos, y_train, k):
    modas = stats.mode(y_train[vecinos[::, :k]], axis=1)[0]
    return modas


def contar_predicciones_correctas(guesses, y_dev):
    correctas = len(guesses) - np.count_nonzero(guesses - y_dev)
    return correctas


def medir_exactitud(vecinos, y_train, y_dev, k) -> float:
    predicciones = moda_cada_elemento(vecinos, y_train, k)
    correctas = contar_predicciones_correctas(predicciones, y_dev)
    totales = len(y_dev)
    # print(predicciones)
    # print(correctas)
    return correctas / totales


def normalize_data(train_data):
    train_normas = np.diag(1 / np.linalg.norm(train_data, axis=1))
    train_normalized = train_normas @ train_data
    return train_normalized
