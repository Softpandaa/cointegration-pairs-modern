"""Point-in-time universe: S&P 500 membership spells mapped to Yahoo price series.

Writes results/universe.csv, one row per spell with status ok, no_data, reused (the ticker now
belongs to another security), partial, excluded (data/exclude.csv) or duplicate (overlaps another
usable spell on the same symbol).
"""
from itertools import pairwise

import pandas as pd

from src.config import DATA, MIN_COVER, PRICE_SPAN, RESULTS, START_SLACK_DAYS


def _split(sp, tmap, ex):
    """Apply the rename map; dated renames and exclusion `until` dates split a spell."""
    out = []
    for r in sp.itertuples():
        rules = tmap[tmap.ticker == r.ticker]
        dates = pd.concat([rules["from"], rules["to"], ex.loc[ex.ticker == r.ticker, "until"]])
        end = r.end if pd.notna(r.end) else pd.Timestamp.max
        cuts = sorted({pd.Timestamp(x) for x in dates.dropna() if r.start < pd.Timestamp(x) < end})
        bounds = [r.start, *cuts, r.end]
        for a, b in pairwise(bounds):
            sym = r.ticker
            for q in rules.itertuples():
                lo = pd.Timestamp(q[3]) if pd.notna(q[3]) else pd.Timestamp.min
                hi = pd.Timestamp(q[4]) if pd.notna(q[4]) else pd.Timestamp.max
                if lo <= a < hi:
                    sym = q.symbol
            out.append((r.ticker, a, b, sym))
    return pd.DataFrame(out, columns=["ticker", "start", "end", "symbol"])


def build():
    sp = pd.read_csv(DATA / "membership_spells.csv", parse_dates=["start", "end"])
    tmap = pd.read_csv(DATA / "ticker_map.csv", dtype=str)
    ex = pd.read_csv(DATA / "exclude.csv", dtype=str)
    sec = pd.read_csv(DATA / "sectors.csv").set_index("Ticker").Sector
    fill = pd.read_csv(DATA / "sector_fill.csv").set_index("symbol").sector
    sec = pd.concat([sec[~sec.index.isin(fill.index)], fill])
    px = pd.read_csv(DATA / "prices.csv.gz", usecols=["Date", "Ticker"], parse_dates=["Date"])
    cal = pd.DatetimeIndex(sorted(px.Date.unique()))
    dates = px.groupby("Ticker").Date.apply(pd.DatetimeIndex)
    first, last = (pd.Timestamp(d) for d in PRICE_SPAN)

    rows = []
    for r in _split(sp, tmap, ex).itertuples():
        a = max(r.start, first)
        b = min(r.end, last) if pd.notna(r.end) else last
        need = ((cal >= a) & (cal < b)).sum()
        if need == 0:
            continue
        day = r.start.strftime("%Y-%m-%d")
        hit = ex[(ex.ticker == r.ticker) & (ex.start.isna() | (ex.start == day))
                 & (ex.until.isna() | (pd.to_datetime(ex.until) > r.start))]
        if len(hit):
            status = "excluded"
        elif r.symbol not in dates.index:
            status = "no_data"
        else:
            d = dates[r.symbol]
            if ((d >= a) & (d < b)).sum() / need >= MIN_COVER:
                status = "ok"
            elif d.min() > a + pd.Timedelta(days=START_SLACK_DAYS):
                status = "reused"
            else:
                status = "partial"
        rows.append((r.ticker, r.start, r.end, r.symbol, sec.get(r.symbol, ""), status))
    u = pd.DataFrame(rows, columns=["ticker", "start", "end", "symbol", "sector", "status"])
    for _, g in u[u.status == "ok"].groupby("symbol"):
        g = g.sort_values("start")
        clash = g.start.values[1:] < g.end.fillna(pd.Timestamp.max).values[:-1]
        u.loc[g.index[1:][clash], "status"] = "duplicate"
    RESULTS.mkdir(exist_ok=True)
    u.to_csv(RESULTS / "universe.csv", index=False, date_format="%Y-%m-%d")
