"""Combination methods of Table 1: weights normalized by (9) and the formation statistic."""
import warnings

import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.config import J0

# Johansen critical values are unused; only the trace statistic is.
warnings.filterwarnings("ignore", message="Critical values are only available")
# numpy >= 2.5 returns eigenvalues as complex with zero imaginary parts.
warnings.filterwarnings("ignore", message="Casting complex values to real")


def adf_t(E):
    """ADF tau of equation (5) for each column of E."""
    E = np.asarray(E, float)
    if E.ndim == 1:
        return adf_t(E[:, None])[0]
    d = np.diff(E, axis=0)
    y, lev, lag = d[1:], E[1:-1], d[:-1]
    n = y.shape[0]
    Z = np.stack([np.ones_like(lev), lev, lag], axis=2)
    ZZ = np.einsum("tni,tnj->nij", Z, Z)
    Zy = np.einsum("tni,tn->ni", Z, y)
    b = np.linalg.solve(ZZ, Zy[..., None])[..., 0]
    u = y - np.einsum("tni,ni->tn", Z, b)
    s2 = (u ** 2).sum(0) / (n - 3)
    se = np.sqrt(s2 * np.linalg.inv(ZZ)[:, 1, 1])
    return b[:, 1] / se


def _unit(w):
    """Equation (9), largest weight positive."""
    w = np.asarray(w, float)
    w = w / np.abs(w).sum()
    return w if w[np.argmax(np.abs(w))] > 0 else -w


def c1_pairs(X):
    return c1_index(X, *np.triu_indices(X.shape[1], 1))


def c1_index(X, i, j):
    """Pairs (i[k], j[k]), equation (10). Returns pairs, weights and tau."""
    Xc = X - X.mean(0)
    C = Xc.T @ Xc
    b_ij, b_ji = C[i, j] / C[j, j], C[i, j] / C[i, i]
    t_ij = adf_t(Xc[:, i] - b_ij * Xc[:, j])
    t_ji = adf_t(Xc[:, j] - b_ji * Xc[:, i])
    first = t_ij <= t_ji
    one = np.ones_like(b_ij)
    w = np.where(first[:, None], np.c_[one, -b_ij], np.c_[-b_ji, one])
    w = np.array([_unit(r) for r in w])
    return np.c_[i, j], w, np.minimum(t_ij, t_ji)


def c2_johansen(X):
    """Equations (11) and (12), one lagged difference."""
    res = coint_johansen(X, det_order=0, k_ar_diff=1)
    return _unit(res.evec[:, 0]), res.lr1[0]


def _smallest_eig(M):
    _, vecs = np.linalg.eigh(M)
    return _unit(vecs[:, 0])


def c3_pca(X):
    """Equation (13)."""
    Xc = X - X.mean(0)
    w = _smallest_eig(Xc.T @ Xc / len(X))
    return w, adf_t(X @ w)


def c3b_zry(X):
    """Equation (14)."""
    Xc = X - X.mean(0)
    n = len(X)
    W = np.zeros((X.shape[1],) * 2)
    for j in range(J0 + 1):
        S = Xc[j:].T @ Xc[:n - j] / n
        W += S @ S.T
    w = _smallest_eig(W)
    return w, adf_t(X @ w)


def c4_stock_vs_rest(X):
    """Equation (15), one spread per stock. Returns weights (m, m) and tau."""
    m = X.shape[1]
    rest = (X.sum(1, keepdims=True) - X) / (m - 1)
    xc, rc = X - X.mean(0), rest - rest.mean(0)
    b = (xc * rc).sum(0) / (rc ** 2).sum(0)
    W = np.array([_unit(np.where(np.arange(m) == k, 1.0, -b[k] / (m - 1))) for k in range(m)])
    return W, adf_t(xc - b * rc)
