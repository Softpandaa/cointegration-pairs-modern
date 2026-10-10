"""Every number of the report, and Figure 1.

    python -m src.report

The portfolio (4) of a method averages its cells (G, L) with a top-decile spread in each trading
period. Means are annualized by 12 / H, and t is the Newey-West t of the period series.
"""
import matplotlib
import numpy as np
import pandas as pd

from src.config import (COSTS_BP, EXITS, FIGURE, GROUPINGS, H_GRID, H_MAIN, L_GRID, METHODS,
                        RESULTS, SAMPLES)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"C1": "Pairs", "C2": "Johansen", "C3": "PCA", "C3b": "Eigenanalysis",
         "C4": "Stock versus group"}
CELL = ["C", "G", "L", "H", "t0"]


def nw_t(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    u = x - x.mean()
    lag = int(4 * (n / 100) ** (2 / 9))
    v = (u @ u) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def cells(sample, H=None):
    d = pd.read_csv(RESULTS / sample / "cells.csv.gz")
    return d if H is None else d[d.H == H]


def per_cell(d, cols, decile=0):
    """Mean per spread of `cols` in each (C, G, L, H, t0), for one decile (None: all)."""
    x = d if decile is None else d[d.decile == decile]
    s = x.groupby(CELL)[[*cols, "n"]].sum()
    return s[cols].div(s.n, axis=0)


def portfolio(m):
    """Equation (4), for each method and H."""
    return m.groupby(["C", "H", "t0"]).mean()


def stat(x, H):
    x = x.dropna()
    return 100 * x.mean() * 12 / H, nw_t(x)


def all_methods(m, H):
    """Equal-weighted average of the method portfolios."""
    p = portfolio(m).net.groupby(["C", "t0"]).mean()
    return stat(p.groupby("t0").mean(), H)


def table2(sample):
    d = cells(sample, H_MAIN)
    q = ["tau_T", "null_tau_T", "tau_E", "null_tau_E", "retest", "null_retest"]
    p = portfolio(per_cell(d, q)).groupby("C").mean()
    top = d[d.decile == 0].groupby(["C", "t0"]).n.sum().groupby("C").mean()
    pi0 = np.minimum(1, d[d.decile >= 5].groupby("C").n.sum() / d.groupby("C").n.sum() / 0.5)
    rows = []
    for c in METHODS:
        r = p.loc[c]
        rows.append([NAMES[c], f"{pi0[c]:.2f}", f"{top[c]:.0f}", f"{r.tau_T:.2f}",
                     f"{r.null_tau_T:.2f}", f"{r.tau_E:.2f}", f"{r.null_tau_E:.2f}",
                     f"{100 * r.retest:.1f}", f"{100 * r.null_retest:.1f}"])
    return rows


def table3(sample):
    d = cells(sample, H_MAIN)
    top = portfolio(per_cell(d, ["net", "null_net"]))
    bot = per_cell(d, ["net"], decile=9)
    tb = portfolio((per_cell(d, ["net"]).net - bot.net).dropna().to_frame("tb"))
    rows = []
    for c in METHODS:
        x, y = top.loc[c], tb.loc[c].tb
        n, nt = stat(x.net, H_MAIN)
        lift, lt = stat(x.net - x.null_net, H_MAIN)
        b, bt = stat(y, H_MAIN)
        rows.append([NAMES[c], f"{n:.2f}", f"{nt:.2f}", f"{1200 / H_MAIN * x.null_net.mean():.2f}",
                     f"{lift:.2f}", f"{lt:.2f}", f"{b:.2f}", f"{bt:.2f}"])
    return rows


def table4(sample):
    """Mean and t of each (L, H)."""
    m = per_cell(cells(sample), ["net"]).reset_index()
    return {(L, H): all_methods(m[(m.L == L) & (m.H == H)].set_index(CELL), H)
            for L in L_GRID for H in H_GRID}


def groupings(sample):
    """Each grouping, and each cluster grouping minus sectors."""
    m = per_cell(cells(sample, H_MAIN), ["net"]).reset_index()
    level = {G: all_methods(m[m.G == G].set_index(CELL), H_MAIN) for G in GROUPINGS}
    p = m.set_index(CELL).net.groupby(["C", "G", "t0"]).mean().groupby(["G", "t0"]).mean()
    diff = {G: stat(p.loc[G] - p.loc["G0"], H_MAIN) for G in GROUPINGS[1:]}
    return level, diff


def retest_share(sample):
    """Re-test pass rate (%) of all spreads and of the null, averaged as the portfolios."""
    s = cells(sample, H_MAIN).groupby(["C", "G", "L", "t0"])[["retest", "null_retest", "n"]].sum()
    f = s[["retest", "null_retest"]].div(s.n, axis=0)
    f = f.groupby(["C", "t0"]).mean().groupby("t0").mean()
    return 100 * f.mean()


def clusters(sample):
    """Mean group count and unassigned share of each grouping by L."""
    g = pd.read_csv(RESULTS / sample / "groups.csv.gz")
    noise = g.group.astype(str) == "-1"
    count = g[~noise].groupby(["method", "L", "t0"]).group.nunique()
    count = count.groupby(["method", "L"]).mean()
    share = noise.groupby([g.method, g.L, g.t0]).mean().groupby(["method", "L"]).mean()
    return pd.DataFrame({"clusters": count, "unassigned": share})


def appendix(sample):
    d = cells(sample, H_MAIN)
    v = [c for c in d.columns if c.startswith("net_e")]
    m = per_cell(d, v)
    return {c: all_methods(m[[c]].rename(columns={c: "net"}), H_MAIN) for c in v}


def periods():
    """Top-decile net return of each method portfolio by trading period, both samples."""
    m = pd.concat([portfolio(per_cell(cells(s, H_MAIN), ["net"])) for s in SAMPLES])
    return m.net.unstack("C").droplevel("H")


def figure(r):
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
    dashes = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
    end = pd.to_datetime(r.index) + pd.DateOffset(months=H_MAIN)
    x = [pd.Timestamp(SAMPLES["train"][0]), *end]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for c, col, ls in zip(METHODS, colors, dashes):
        y = np.r_[0, 100 * ((1 + r[c].to_numpy()).cumprod() - 1)]
        ax.plot(x, y, color=col, linestyle=ls, linewidth=1.2, label=NAMES[c])
    ax.axvline(pd.Timestamp(SAMPLES["test"][0]), color="gray", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Cumulative net return (%)")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=2.5)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")


def row(items):
    return " & ".join(items) + r" \\"


if __name__ == "__main__":
    for s in SAMPLES:
        print(f"Section 3.1 clusters, {s}\n{clusters(s).round(2).to_string()}")
    for table in (table2, table3):
        for s in SAMPLES:
            print(f"{table.__name__} {s}")
            print("\n".join(row(r) for r in table(s)))
    for s in SAMPLES:
        print(f"Re-test pass (%) of all spreads, {s}:", retest_share(s).round(2).to_dict())
    r = periods()
    figure(r)
    first = r[r.index >= SAMPLES["test"][0]].iloc[0]
    print("First half of 2020, mean of the methods (%):", round(100 * first.mean(), 2))
    for s in SAMPLES:
        t4 = table4(s)
        print(f"table4 {s}, mean (t)")
        for L in L_GRID:
            print(row([str(L)] + [f"{t4[L, H][0]:.2f} ({t4[L, H][1]:.2f})" for H in H_GRID]))
        level, diff = groupings(s)
        for name, d in (("Groupings", level), ("Minus sectors", diff)):
            print(f"{name}, mean and t:",
                  {G: [float(round(v, 2)) for v in x] for G, x in d.items()})
    app = {s: appendix(s) for s in SAMPLES}
    print("Appendix B, train then test, mean and t")
    for v in app["train"]:
        print(row([v] + [f"{a:.2f}" for s in SAMPLES for a in app[s][v]]))
    gap = max(abs(app[s][f"net_e{e:+.1f}_c{c}"][0] - app[s][f"net_e+0.0_c{c}"][0])
              for s in SAMPLES for e in EXITS for c in COSTS_BP)
    print("Largest change from the exit at the mean (%):", round(gap, 2))
