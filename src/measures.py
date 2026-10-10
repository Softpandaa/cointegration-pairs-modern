"""Persistence measures and the trading rule (3) on spreads with formation weights held fixed."""
import numpy as np
from statsmodels.tsa.adfvalues import mackinnoncrit

from src.combine import adf_t
from src.config import BORROW, COST_BP, COSTS_BP, ENTRY, EXITS, TRADING_DAYS


def legs(XE, W):
    """Value of the long and short legs over E."""
    return XE @ np.clip(W, 0, None).T, XE @ np.clip(-W, 0, None).T


def zscore(ST, SE):
    return (SE - ST.mean(0)) / ST.std(0, ddof=1)


def trade(Z, S, up, dn, exit=0.0):
    """Rule (3), closing at z n >= -exit. Returns gross gain, units traded, short-leg years."""
    n, N = Z.shape
    pos, gross, traded, short = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    expo = up + dn
    for t in range(n):
        if t:
            gross += pos * (S[t] - S[t - 1])
            short += np.where(pos > 0, dn[t], np.where(pos < 0, up[t], 0.0)) / TRADING_DAYS
        close = (pos != 0) & ((Z[t] * -pos) <= exit)
        if t == n - 1:
            close = pos != 0
        traded += expo[t] * close
        pos = np.where(close, 0.0, pos)
        if t < n - 1:
            enter = (pos == 0) & (np.abs(Z[t]) >= ENTRY)
            traded += expo[t] * enter
            pos = np.where(enter, -np.sign(Z[t]), pos)
    return gross, traded, short


def net(gross, traded, short, bp=COST_BP):
    return gross - bp / 1e4 * traded - BORROW * short


def measures(ST, SE, UP, DN):
    """tau_E, the re-test and the net return of each spread over E."""
    tau_E = adf_t(SE)
    crit = mackinnoncrit(1, "c", nobs=len(SE) - 2)[1]
    return {"tau_E": tau_E, "retest": (tau_E < crit).astype(float),
            "net": net(*trade(zscore(ST, SE), SE, UP, DN))}


def variants(ST, SE, UP, DN):
    """Net return under each exit and cost of Appendix B."""
    Z = zscore(ST, SE)
    parts = {e: trade(Z, SE, UP, DN, e) for e in EXITS}
    return {f"net_e{e:+.1f}_c{c}": net(*parts[e], c) for e in EXITS for c in COSTS_BP}
