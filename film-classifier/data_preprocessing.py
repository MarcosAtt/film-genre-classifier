import pandas as pd
import numpy as np
from pca import *
from variables import *
from cross_validation import *


def import_data():
    return pd.read_csv("../data/raw/wiki_movie_plots_deduped_sample.csv")


def document_term_matrix(df, Q):
    tokens = np.hstack(df["tokens"].apply(lambda x: x.split()).values)
    unique_tokens = pd.Series(tokens).value_counts().index[:Q].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))

    X_train = np.zeros((320, len(unique_tokens)), dtype=int)
    X_test = np.zeros((80, len(unique_tokens)), dtype=int)
    y_train = np.zeros((320, 1), dtype=int)
    y_test = np.zeros((80, 1), dtype=int)
    itrain = 0
    itest = 0

    for i, row in df.iterrows():
        if row["split"] == "train":
            for token in row["tokens"].split():
                if unique_tokens_dict.get(token, False) is not False:
                    X_train[itrain, unique_tokens_dict[token]] += 1
            y_train[itrain] = genre_name.index(row["Genre"])
            itrain += 1
        else:
            for token in row["tokens"].split():
                if unique_tokens_dict.get(token, False) is not False:
                    X_test[itest, unique_tokens_dict[token]] += 1
            y_test[itest] = genre_name.index(row["Genre"])
            itest += 1
    return X_train, y_train, X_test, y_test


def precomputar_folds_covMatEVD():
    """
    Returns V_folds[i] = covMatEvd(X_fold[i]).V
    """
    data = document_term_matrix(import_data(), default_Q)
    X_train = data[0]
    y_train = data[1]
    X_train, y_train = balancear_clases(X_train, y_train)
    V_folds = []
    for i in range(5):
        X_newtrain, X_dev, y_newtrain, y_dev = separate_dev_data(X_train, y_train, i)
        S, V = covarianceMatrixEVD(X_newtrain, 100, 1e-7)
        V_folds.append(V)
    return V_folds


def save_variable(var, filename):
    filename += ".pkl"
    with open(filename, "wb") as f:
        pickle.dump(var, f)


def load_variable(filename):
    """Returns variable from filename.pkl"""
    filename += ".pkl"
    with open(filename, "rb") as f:
        var = pickle.load(f)
    return var


# Save and load covariance matrix to disk
def save_folds_covMatEVD(V_folds):
    save_variable(V_folds, "folds_cov_mat_evd")


def load_folds_covMatEVD():
    """Returns V_folds"""
    try:
        V_folds = load_variable("V_folds")
    except:
        print("No hay archivo con el precalculo de PCA, calculando...")
        V_folds = precomputar_folds_covMatEVD()
        save_variable(V_folds, "V_folds")
    return V_folds


def save_promedios_k_p_exa(promedios_p_k):
    save_variable(promedios_p_k, "promedios_p_k_exactitud")
