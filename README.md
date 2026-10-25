# Do Cointegration-based Pairs Remain Profitable?

This study examines the out-of-sample profitability of cointegration-based pair formation in the S&P 500 from 2010 to 2025, and whether it depends on the combination method, the stock grouping or the window length. Spreads are formed by five methods (pairs, Johansen, principal components, eigenanalysis and stock versus group) within sectors and two kinds of correlation-based clusters, and traded by the 2-standard-deviation threshold rule of Gatev, Goetzmann and Rouwenhorst (2006).

## Findings

No formation choice can consistently generate profit on its own. In the testing period 2020 to 2025, the top decile of every method loses between 0.70% and 1.34% a year. The formation p-value does not consistently rank returns, and neither the grouping nor the window changes the sign. The mean reversion selected at formation persists little more than that of spreads formed on random walks.

## Layout

```
data/     committed membership with sectors, and prices
src/      analysis modules, every parameter declared once in config.py
report.pdf
slides.pdf
```

The report and slides are distributed as compiled PDFs. Their typesetting source is not included.

## Data

`data/members.csv` holds the point-in-time [S&P 500 membership](https://github.com/fja05680/sp500) mapped to Yahoo Finance symbols, one row per membership spell with usable prices, with the sector of the stock. `data/prices.csv.gz` holds the daily adjusted closing prices of these stocks from Yahoo Finance, 2005 to 2025.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python -m src.run
python -m src.report
```

`src.run` forms and evaluates the spreads of both samples, and `src.report` prints every number quoted in the report and writes its figure to `latex/cumulative.png`, which is created on the first run and is not tracked.
