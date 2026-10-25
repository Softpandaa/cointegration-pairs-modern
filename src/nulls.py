"""Random walk null (2): each method on m independent random walks of T formation days.

A cell (method, m, T) holds the sorted null statistics, which give the p-value of a spread, and
the mean null measures by H and p-value decile, from the same paths extended over one year.
"""
from functools import cache
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.combine import adf_t, c1_pairs, c2_johansen, c3_pca, c3b_zry, c4_stock_vs_rest
from src.config import (BASKET, CHUNK, DATA, DAYS_PER_STOCK, DRAWS, DRAWS_C1, H_GRID, L_GRID,
                        MONTH_DAYS, NULL_CELLS, SIGMA_SPAN, TRADING_DAYS)
from src.measures import legs, measures

T_GRID = [TRADING_DAYS * L for L in L_GRID]


@cache
def sigma_pool():
    """Daily log return s.d. of each usable stock with at least a year of returns."""
    px = pd.read_csv(DATA / "prices.csv.gz", parse_dates=["Date"])
    px = px[(px.Date >= SIGMA_SPAN[0]) & (px.Date < SIGMA_SPAN[1])]
    px = px.sort_values(["Ticker", "Date"])
    r = np.log(px.Close).groupby(px.Ticker).diff()
    g = r.groupby(px.Ticker)
    return g.std()[g.count() >= TRADING_DAYS].to_numpy()


def simulate(m, T, sig, rng, extra=0):
    """Normalized prices over T formation days and `extra` days after."""
    x = np.cumsum(rng.choice(sig, m) * rng.standard_normal((T + extra, m)), axis=0)
    return np.exp(x - x[T - 1])


def candidates(method, X):
    """Weights (N, m) and statistics (N,) of one method on one group."""
    m = X.shape[1]
    if method == "C1":
        pr, w, st = c1_pairs(X)
        W = np.zeros((len(pr), m))
        W[np.arange(len(pr))[:, None], pr] = w
        return W, st
    if method == "C4":
        return c4_stock_vs_rest(X)
    w, st = {"C2": c2_johansen, "C3": c3_pca, "C3b": c3b_zry}[method](X)
    return w[None, :], np.array([st])


def pvalue(null, method, s):
    if method == "C2":                              # trace statistic, large is evidence
        return (len(null) - np.searchsorted(null, s, side="left")) / len(null)
    return np.searchsorted(null, s, side="right") / len(null)


def decile(p):
    return np.minimum((p * 10).astype(int), 9)


def key(method, m, T):
    return f"{method}_{2 if method == 'C1' else m}_{min(T_GRID, key=lambda g: abs(g - T))}"


def null_cell(c, m, T):
    seeds = [[0, sum(map(ord, c)), m, T, d] for d in range(DRAWS_C1 if c == "C1" else DRAWS)]
    sig = sigma_pool()
    fits = [candidates(c, simulate(m, T, sig, np.random.default_rng(s))) for s in seeds]
    null = np.sort(np.concatenate([f[1] for f in fits]))
    acc, buf, size = {}, [], 0

    def flush():
        ST, SE, UP, DN = (np.concatenate([b[i] for b in buf], 1) for i in range(4))
        dec = np.concatenate([b[4] for b in buf])
        tau_T = adf_t(ST)
        for H in H_GRID:
            n = MONTH_DAYS * H
            out = {"tau_T": tau_T, **measures(ST, SE[:n], UP[:n], DN[:n])}
            for d in range(10):
                k = dec == d
                acc.setdefault((H, d), []).append({q: v[k] for q, v in out.items()})
        buf.clear()

    for s, (W, st) in zip(seeds, fits):
        X = simulate(m, T, sig, np.random.default_rng(s), extra=MONTH_DAYS * max(H_GRID))
        buf.append((X[:T] @ W.T, X[T:] @ W.T, *legs(X[T:], W), decile(pvalue(null, c, st))))
        size += len(st)
        if size >= CHUNK:
            flush()
            size = 0
    if buf:
        flush()
    rows = []
    for (H, d), parts in acc.items():
        v = {q: np.concatenate([x[q] for x in parts]) for q in parts[0]}
        r = dict(method=c, m=m, T=T, H=H, decile=d, n=len(v["net"]))
        for q, a in v.items():
            r[q] = np.nanmean(a) if np.isfinite(a).any() else np.nan
        rows.append(r)
    return null, rows


def _build_one(cell):
    f = NULL_CELLS / f"{key(*cell)}.npz"
    if not f.exists():
        null, rows = null_cell(*cell)
        np.savez_compressed(f, null=null, rows=pd.DataFrame(rows).to_records(index=False))


def build(cells, procs=2):
    """Simulate every cell not yet in results/null_cells."""
    NULL_CELLS.mkdir(parents=True, exist_ok=True)
    todo = sorted(set(cells), key=lambda x: -x[1] * x[2])
    with Pool(procs) as pool:
        for k, _ in enumerate(pool.imap_unordered(_build_one, todo), 1):
            if k % 50 == 0:
                print(f"null cells: {k} of {len(todo)}", flush=True)


def load():
    """Null statistics by cell key, and mean null measures indexed by (H, key, decile)."""
    table, rows = {}, []
    for f in sorted(NULL_CELLS.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        table[f.stem] = z["null"]
        rows.append(pd.DataFrame(z["rows"]).assign(key=f.stem))
    return table, pd.concat(rows).set_index(["H", "key", "decile"]).sort_index()


def methods(m, T):
    """Methods other than pairs for a group of m stocks over T days."""
    if m < 3:
        return []
    return ["C4"] + [c for c in sorted(BASKET) if T / m >= DAYS_PER_STOCK]


def cells_needed(groups):
    """(method, m, T) of every group."""
    d = pd.read_csv(groups)
    d = d[d.group.astype(str) != "-1"]
    s = d.groupby(["L", "method", "t0", "group"]).size().reset_index(name="m")
    cells = [("C1", 2, TRADING_DAYS * L) for L in sorted(s.L.unique())]
    for L, m in s[["L", "m"]].drop_duplicates().itertuples(index=False):
        T = TRADING_DAYS * L
        cells += [(c, m, T) for c in methods(m, T)]
    return cells
