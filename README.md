# Do Cointegration-based Pairs Remain Profitable?

This study tests whether cointegration-based pair formation remains profitable out of sample in the S&P 500, 2010 to 2025. Spreads are formed by five methods (pairs, Johansen, principal components, eigenanalysis and stock versus group) within sectors and correlation-based clusters over 1- to 5-year windows, and traded by the 2-standard-deviation rule of Gatev, Goetzmann and Rouwenhorst (2006).

## Findings

No formation choice is profitable on its own. From 2020 to 2025, the top decile of every method loses between 0.70% and 1.34% a year after costs, and neither the grouping nor the window changes the sign.

The formation p-value does not consistently rank later returns, and the mean reversion found at formation persists little more than in spreads formed on independent random walks.

## Layout

```
data/     index membership with sectors, and daily prices
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/members.csv` lists the point-in-time [S&P 500 membership](https://github.com/fja05680/sp500), one row per membership spell with usable prices, with the Yahoo Finance symbol and sector of each stock. `data/prices.csv.gz` holds their daily adjusted closing prices from Yahoo Finance, 2005 to 2025.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python -m src.run
python -m src.report
```

`src.run` forms and evaluates the spreads of both samples. `src.report` then prints every number quoted in the report and draws its figure.
