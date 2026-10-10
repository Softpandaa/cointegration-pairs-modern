"""Spreads of every grouping and method by formation window, summed by p-value decile.

Each row (G, C, decile, L, H, t0) holds the spread count n, the sums of the measures and the
sums of their null values for the same (C, m, T, H, decile).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import grouping, nulls
from src.combine import adf_t, c1_index
from src.config import H_GRID, H_MAIN
from src.measures import legs, measures, variants

NULLQ = ["tau_T", "tau_E", "retest", "net"]
_state = {}


def h_starting(t0):
    """Horizons whose trading windows start at t0, aligned to January."""
    return [H for H in H_GRID if (t0.month - 1) % H == 0]


def group_lists(g, syms):
    """Column indices of each group with at least 2 stocks, by grouping."""
    pos = {s: k for k, s in enumerate(syms)}
    out = {}
    for G, x in g.groupby("method"):
        x = x[x.group.astype(str) != "-1"]
        members = x.groupby("group").symbol.apply(list)
        out[G] = [np.array([pos[s] for s in v]) for v in members if len(v) >= 2]
    return out


def window(px, t0, L, syms):
    """Prices normalized by the last formation day; a delisted stock keeps its last price."""
    F = px.loc[(px.index >= t0 - pd.DateOffset(years=L)) & (px.index < t0), syms]
    E = px.loc[(px.index >= t0) & (px.index < t0 + pd.DateOffset(months=max(H_GRID))), syms]
    E = pd.concat([F.iloc[[-1]], E]).ffill().iloc[1:]
    p0 = F.iloc[-1].to_numpy()
    return F.to_numpy() / p0, E.to_numpy() / p0, E.index


def candidate_set(XT, groups):
    """Pairs of all groupings with their membership, and the other spreads."""
    pairs = {G: {(min(a, b), max(a, b))
                 for idx in gl for a, b in zip(*np.triu_indices(len(idx), 1))
                 for a, b in [(idx[a], idx[b])]}
             for G, gl in groups.items()}
    union = sorted(set().union(*pairs.values()))
    i, j = (np.array(v) for v in zip(*union))
    pr, w, st = c1_index(XT, i, j)
    W1 = np.zeros((len(pr), XT.shape[1]))
    W1[np.arange(len(pr))[:, None], pr] = w
    member = {G: np.array([u in pairs[G] for u in union]) for G in groups}
    T, other = len(XT), []
    for G, gl in groups.items():
        for idx in gl:
            m = len(idx)
            for c in nulls.methods(m, T):
                Wg, sg = nulls.candidates(c, XT[:, idx])
                Wf = np.zeros((len(sg), XT.shape[1]))
                Wf[:, idx] = Wg
                other.append((G, c, m, Wf, sg))
    return (W1, st, member), other


def evaluate(L, t0, g):
    syms = sorted(g[g.method == "G0"].symbol)
    XT, XE, edates = window(_state["px"], t0, L, syms)
    T = len(XT)
    (W1, st1, member), other = candidate_set(XT, group_lists(g, syms))
    Wall = np.vstack([W1] + [o[3] for o in other])
    tag = pd.DataFrame({"C": ["C1"] * len(st1) + [o[1] for o in other for _ in o[4]],
                        "m": [2] * len(st1) + [o[2] for o in other for _ in o[4]],
                        "stat": np.concatenate([st1] + [o[4] for o in other])})
    tag["key"] = [nulls.key(c, m, T) for c, m in zip(tag.C, tag.m)]
    tag["decile"] = nulls.decile(np.array([nulls.pvalue(_state["table"][k], c, s)
                                           for k, c, s in zip(tag.key, tag.C, tag.stat)]))
    ST = XT @ Wall.T
    tau_T = adf_t(ST)
    rows = []
    for H in h_starting(t0):
        n = int((edates < t0 + pd.DateOffset(months=H)).sum())
        SE = XE[:n] @ Wall.T
        UP, DN = legs(XE[:n], Wall)
        meas = {"tau_T": tau_T, **measures(ST, SE, UP, DN)}
        if H == H_MAIN:
            meas |= variants(ST, SE, UP, DN)
        base = pd.concat([tag, pd.DataFrame(meas)], axis=1)
        null = _state["nm"].loc[H].reindex(pd.MultiIndex.from_arrays([base.key, base.decile]))
        for q in NULLQ:
            base[f"null_{q}"] = null[q].to_numpy()
        c1 = base.iloc[:len(st1)]
        rest = base.iloc[len(st1):].assign(G=[o[0] for o in other for _ in o[4]])
        df = pd.concat([c1[mask].assign(G=G) for G, mask in member.items()] + [rest])
        cols = list(meas) + [f"null_{q}" for q in NULLQ]
        s = df.groupby(["G", "C", "decile"])
        agg = s[cols].sum(min_count=1)
        agg["n"] = s.size()
        rows.append(agg.reset_index().assign(L=L, H=H, t0=t0.date()))
    return rows


def _init():
    _state["px"] = grouping.load()[1]
    _state["table"], _state["nm"] = nulls.load()


def run(groups, path, procs=2):
    g = pd.read_csv(groups, parse_dates=["t0"])
    jobs = [(L, t0, x) for (L, t0), x in g.groupby(["L", "t0"])]
    with Pool(procs, _init) as pool:
        rows = [r for rs in pool.starmap(evaluate, jobs) for r in rs]
    pd.concat(rows).to_csv(path, index=False, compression="gzip")
