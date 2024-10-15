import pandas as pd
import numpy as np
from variables import *


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
