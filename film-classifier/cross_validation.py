from knn import *
import numpy as np


def print_avance(i, j, iMax, jMax):
    total = (iMax) * (jMax)
    porcentaje = (i * jMax) + (j)
    if i * j < total:
        print(int(porcentaje / total * 100), "%", end="\r", flush=True)
    if (i + 1) * (j + 1) == total:
        print("100 %!")


def separate_dev_data(X_train, y_train, i):
    part = X_train.shape[0] // 4  # Partition size
    X_dev = X_train[part * i : part * (i + 1)]
    y_dev = y_train[part * i : part * (i + 1)]

    X_newtrain = np.concatenate((X_train[0 : part * i], X_train[part * (i + 1) : :]))
    y_newtrain = np.concatenate((y_train[0 : part * i], y_train[part * (i + 1) : :]))
    return X_newtrain, X_dev, y_newtrain, y_dev


def four_fold_cross_validation_k_exploration(X_train, y_train, maxK):
    """
    Knn para el rango 1,maxK. Devuelve promedios de hacer 4-fold cross-validation
    """
    promedio_aciertos_k = np.zeros(maxK)
    for i in range(4):  # FOLD
        X_newtrain, X_dev, y_newtrain, y_dev = separate_dev_data(X_train, y_train, i)

        X_newtrain_normalized = normalize_data(X_newtrain)
        X_dev_normalized = normalize_data(X_dev)

        vecinos = calcular_vecinos(X_newtrain_normalized, X_dev_normalized)

        for k in range(1, maxK):
            # print_avance(i, k, 4, maxK + 1)
            promedio_aciertos_k[k] += medir_exactitud(vecinos, y_newtrain, y_dev, k)

    for k in range(1, maxK):
        promedio_aciertos_k[k] /= 4

    return promedio_aciertos_k
