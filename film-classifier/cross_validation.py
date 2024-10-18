from knn import *
import numpy as np
from pca import *


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


def balancear_clases(a, b):
    """Reordenar ambos arrays de la misma manera"""
    p = np.random.permutation(len(a))
    return a[p], b[p]


def four_fold_cross_validation_k_exploration(X_train, y_train, maxK):
    """
    Knn para el rango 1,maxK. Devuelve promedios de hacer 4-fold cross-validation
    """
    promedio_aciertos_k = np.zeros(maxK)
    X_train, y_train = balancear_clases(X_train, y_train)
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


def five_fold_cross_validation_k_p_exploration(V_folds, X_train, y_train, maxK, ps):
    """
    V_folds: autovectores de la matriz de covarianza de cada fold de X_train,
    X_train, y_train: datos y clase correspondiente de entrenamiento,
    maxK: maxima cantidad de vecinos,
    ps: secuencia con la cantidad de componentes principales a probar

    devuelve % aciertos [todos los p de ps][0 a maxK]
    """
    l_ps = len(ps)
    promedio_exactitud_p_k = np.zeros((l_ps, maxK))
    X_train, y_train = balancear_clases(X_train, y_train)

    for i in range(4):  # i fold
        X_newtrain, X_dev, y_newtrain, y_dev = separate_dev_data(X_train, y_train, i)
        X_newtrain_normalized = normalize_data(X_newtrain)
        X_dev_normalized = normalize_data(X_dev)

        for j, p in enumerate(ps):  # p components
            print_avance(i, j, 4, l_ps)

            pca_newtrain = reduce_dimensionality(X_newtrain_normalized, V_folds[i], p)
            pca_dev = reduce_dimensionality(X_dev_normalized, V_folds[i], p)
            vecinos = calcular_vecinos(pca_newtrain, pca_dev)

            for k in range(1, maxK):  # k neighbors
                promedio_exactitud_p_k[j][k] += medir_exactitud(
                    vecinos, y_newtrain, y_dev, k
                )

    for i in range(l_ps):
        for k in range(1, maxK):
            promedio_exactitud_p_k[i][k] /= 4

    return promedio_exactitud_p_k


def arg_max_p_k(ps, k_maximo, promedios_p_k):
    """
    ps: lista de p,
    k_maximo: numero,
    promedios_p_k: matriz 2 dimensiones con exactitudes halladas
    """
    mejorExactitud = 0
    P_optimo = 0
    K_optimo = 0

    for i, p in enumerate(ps):
        for k in range(1, k_maximo):
            if promedios_p_k[i][k] > mejorExactitud:
                mejorExactitud = promedios_p_k[i][k]
                P_optimo = p
                K_optimo = k
    return P_optimo, K_optimo, mejorExactitud
