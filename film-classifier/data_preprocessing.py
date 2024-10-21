import pandas as pd
import numpy as np
from pca import *  # noqa: F403
from variables import *  # noqa: F403
import pickle
import math


def import_data():
    return pd.read_csv("../data/raw/wiki_movie_plots_deduped_sample.csv")


def balancear_clases(a, b):
    """Reordenar ambos arrays de la misma manera"""
    p = np.random.default_rng(seed=default_seed).permutation(len(a))
    return a[p], b[p]


def separate_test_data(df):
    df = df.filter(items=["Genre", "tokens", "split"])
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    return df_train, df_test


def separate_dev_data(X_train, y_train, i):
    part = X_train.shape[0] // 4  # Partition size
    X_dev = X_train[part * i : part * (i + 1)]
    y_dev = y_train[part * i : part * (i + 1)]

    X_newtrain = np.concatenate((X_train[0 : part * i], X_train[part * (i + 1) : :]))
    y_newtrain = np.concatenate((y_train[0 : part * i], y_train[part * (i + 1) : :]))
    return X_newtrain, X_dev, y_newtrain, y_dev


def vocabulary(df, Q):
    tokens = np.hstack(df["tokens"].apply(lambda x: x.split()).values)
    unique_tokens = pd.Series(tokens).value_counts().index[:Q].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))
    return unique_tokens_dict


def document_term_matrix(df, Q):
    """Defined as raw count of tokens"""
    unique_tokens_dict = vocabulary(df, Q)
    unique_t_number = len(unique_tokens_dict.keys())

    X = np.zeros((len(df), unique_t_number), dtype=int)  # Term frequency per document
    y = np.zeros((len(df), 1), dtype=int)  # Document genre
    index = 0
    for i, row in df.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token, False) is not False:
                X[index, unique_tokens_dict[token]] += 1
        y[index] = genre_name.index(row["Genre"])
        index += 1
    return X, y


def test_document_term_matrix(df_test, df_train, Q):
    """Defined as raw count of tokens"""
    unique_tokens_dict = vocabulary(df_train, Q)
    unique_t_number = len(unique_tokens_dict.keys())
    X = np.zeros(
        (len(df_test), unique_t_number), dtype=int
    )  # Term frequency per document
    y = np.zeros((len(df_test), 1), dtype=int)  # Document genre
    index = 0
    for i, row in df_test.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token):
                X[index, unique_tokens_dict[token]] += 1
        y[index] = genre_name.index(row["Genre"])
        index += 1
    return X, y


def inverse_document_frequency_matrix(df, Q):
    N = len(df)  # Number of documents
    """Defined as log ((|df| + 1) / (t in d : d in df))"""
    unique_tokens_dict = vocabulary(df, Q)
    unique_t_number = len(unique_tokens_dict.keys())

    term_aparitions = np.zeros(
        (len(df), unique_t_number), dtype=int
    )  # Term frequency per document
    index = 0
    for i, row in df.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token):
                term_aparitions[index, unique_tokens_dict[token]] += 1
        index += 1

    term_aparitions[term_aparitions > 0] = 1
    term_aparitions = term_aparitions.sum(axis=0)

    idf = np.zeros((unique_t_number, 1), dtype=float)
    for i in range(len(term_aparitions)):
        idf[i] = math.log((N + 1) / (term_aparitions[i] + 1))

    return idf


def tf_idf(df, Q):
    pass


def precomputar_folds_covMatEVD(X_train, y_train):
    """
    Returns V_folds[i] = covMatEvd(X_fold[i]).V
    """
    V_folds = []
    for i in range(5):
        print_avance(i, 0, 5, 1)
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
