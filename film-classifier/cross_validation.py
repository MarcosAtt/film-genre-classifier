import knn
import numpy as np


def separate_dev_data(X_train, y_train, i):
    part = X_train.shape[0] // 5  # Partition size
    X_dev = X_train[part * i : part * (i + 1)]
    y_dev = y_train[part * i : part * (i + 1)]

    X_newtrain = np.concatenate((X_train[0 : part * i], X_train[part * (i + 1) : :]))
    y_newtrain = np.concatenate((y_train[0 : part * i], y_train[part * (i + 1) : :]))
    return X_newtrain, X_dev, y_newtrain, y_dev
