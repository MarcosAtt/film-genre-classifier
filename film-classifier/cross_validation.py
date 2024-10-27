import numpy as np

from data_preprocessing import *  # noqa: F403
from knn import *  # noqa: F403
from pca import *  # noqa: F403
from variables import *  # noqa: F403


def four_fold_cross_validation_k_exploration(X_train, y_train, maxK, dist_cos=True):
    """
    Knn para el rango 1,maxK. Devuelve promedios de hacer 4-fold cross-validation
    """
    promedio_aciertos_k = np.zeros(maxK)

    for i in range(4):  # FOLD
        X_newtrain, X_dev, y_newtrain, y_dev = separate_dev_data(X_train, y_train, i)

        X_newtrain_normalized = normalize_data(X_newtrain)
        X_dev_normalized = normalize_data(X_dev)

        vecinos = calcular_vecinos(X_newtrain_normalized, X_dev_normalized, dist_cos)

        for k in range(1, maxK):
            # print_avance(i, k, 4, maxK + 1)
            promedio_aciertos_k[k] += medir_exactitud(vecinos, y_newtrain, y_dev, k)

    for k in range(1, maxK):
        promedio_aciertos_k[k] /= 4

    return promedio_aciertos_k


def four_fold_cross_validation_k_p_exploration(
    V_folds, X_train, y_train, maxK, ps, dist_cos=True
):
    """
    V_folds: autovectores de la matriz de covarianza de cada fold de X_train,
    X_train, y_train: datos y clase correspondiente de entrenamiento,
    maxK: maxima cantidad de vecinos,
    ps: secuencia con la cantidad de componentes principales a probar

    devuelve % aciertos [todos los p de ps][0 a maxK]
    """
    l_ps = len(ps)
    promedio_exactitud_p_k = np.zeros((l_ps, maxK))

    for i in range(4):  # i fold
        X_newtrain, X_dev, y_newtrain, y_dev = separate_dev_data(X_train, y_train, i)
        X_newtrain_normalized = normalize_data(X_newtrain)
        X_dev_normalized = normalize_data(X_dev)

        for j, p in enumerate(ps):  # p components
            print_avance(i, j, 4, l_ps)

            pca_newtrain = reduce_dimensionality(X_newtrain_normalized, V_folds[i], p)
            pca_dev = reduce_dimensionality(X_dev_normalized, V_folds[i], p)
            vecinos = calcular_vecinos(pca_newtrain, pca_dev, dist_cos)

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
