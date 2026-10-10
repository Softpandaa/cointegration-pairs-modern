"""Groupings G0 (sectors), G1 (OPTICS) and G2 (complete linkage), Appendix A.2."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.cluster import OPTICS

from src.config import DATA, L_GRID, MIN_SAMPLES, RESULTS


def load():
    u = pd.read_csv(RESULTS / "universe.csv", parse_dates=["start", "end"])
    ok = u[u.status == "ok"]
    px = pd.read_csv(DATA / "prices.csv.gz", usecols=["Date", "Ticker", "Close"],
                     parse_dates=["Date"])
    px = px[px.Ticker.isin(ok.symbol)]
    px = px.pivot_table(index="Date", columns="Ticker", values="Close").sort_index()
    ex = pd.read_csv(DATA / "exclude.csv").dropna(subset=["until"])
    t2s = u.drop_duplicates("ticker").set_index("ticker").symbol
    valid_from = pd.to_datetime(ex.set_index(ex.ticker.map(t2s)).until).groupby(level=0).max()
    sector = ok.drop_duplicates("symbol").set_index("symbol").sector
    return ok, px, valid_from, sector


def formation_dates(px, span):
    """First trading day of each month."""
    d = px.index[(px.index >= span[0]) & (px.index <= span[1])]
    return pd.DatetimeIndex(pd.Series(d, index=d).groupby(d.to_period("M")).min().values)


def eligible(ok, px, valid_from, t0, L):
    """Prices over the formation window of the members at t0 with a complete record."""
    start = t0 - pd.DateOffset(years=L)
    win = px.loc[(px.index >= start) & (px.index < t0)]
    members = ok[(ok.start <= t0) & (ok.end.isna() | (ok.end > t0))].symbol.unique()
    full = win.columns[win.notna().all()]
    syms = [s for s in members
            if s in full and not (s in valid_from.index and valid_from[s] > start)]
    return win[sorted(syms)]


def residuals(win):
    """eta of equation (6)."""
    r = np.log(win).diff().iloc[1:]
    I = r.mean(axis=1).to_numpy()  # noqa: E741
    X = np.c_[np.ones_like(I), I]
    b = np.linalg.lstsq(X, r.to_numpy(), rcond=None)[0]
    return pd.DataFrame(r.to_numpy() - X @ b, index=r.index, columns=r.columns)


def _lambda_plus(n, T):
    return (1 + np.sqrt(n / T)) ** 2


def g1(c):
    """OPTICS labels, -1 unassigned."""
    T, n = c.shape
    vals, vecs = np.linalg.eigh(np.corrcoef(c.to_numpy(), rowvar=False))
    k = vals > _lambda_plus(n, T)
    if not k.any() or n < MIN_SAMPLES:
        return np.zeros(n, int)
    return OPTICS(min_samples=MIN_SAMPLES).fit(vecs[:, k]).labels_


def g2(c):
    T, n = c.shape
    if n < 2:
        return np.zeros(n, int)
    rho = np.corrcoef(np.apply_along_axis(rankdata, 0, c.to_numpy()), rowvar=False)
    K = max(1, int((np.linalg.eigvalsh(rho) > _lambda_plus(n, T)).sum()))
    d = np.sqrt(np.clip(2 * (1 - rho), 0, None))
    np.fill_diagonal(d, 0)
    return fcluster(linkage(squareform(d, checks=False), "complete"), K, "maxclust") - 1


def groups(win, sector):
    """Rows (grouping, symbol, group label) of one formation window."""
    c = residuals(win)
    sec = sector[c.columns]
    rows = [("G0", s, sec[s]) for s in c.columns]
    for name, f in (("G1", g1), ("G2", g2)):
        for sname, cols in sec.groupby(sec).groups.items():
            lab = f(c[cols])
            rows += [(name, s, f"{sname}|{k}" if k >= 0 else "-1") for s, k in zip(cols, lab)]
    return rows


def run(span, path):
    ok, px, valid_from, sector = load()
    out = []
    for L in L_GRID:
        for t0 in formation_dates(px, span):
            win = eligible(ok, px, valid_from, t0, L)
            out += [(L, t0.date(), *r) for r in groups(win, sector)]
    df = pd.DataFrame(out, columns=["L", "t0", "method", "symbol", "group"])
    df.to_csv(path, index=False, compression="gzip")
