import numpy as np
import eigenvalues


def covarianceMatrixEVD(X, niter, eps):
    """
    Input: X: data_set, niter: iterations, eps: precision
    Returns S: vector of eigenvalues, V: eigenvectors, VSVt = C: covariance matrix
    """
    n = X.shape[0]
    X_centered = X - X.mean(axis=0)
    C = X_centered.T @ X_centered / (n - 1)
    return eigenvalues.decomposition(C, niter, eps)


def reduce_dimensionality(X, V, p):
    """Reduce X to p dimensions using V principal components"""
    return (X @ V)[::, 0:p]
