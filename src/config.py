"""Parameters of the study, each declared once."""
from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
NULL_CELLS = RESULTS / "null_cells"     # one file per (method, m, T)
FIGURE = ROOT / "latex" / "cumulative.png"

# Sample ----------------------------------------------------------------------
SAMPLES = {"train": ("2010-01-01", "2019-12-31"), "test": ("2020-01-01", "2025-12-31")}
SIGMA_SPAN = ("2005-01-01", "2020-01-01")   # sigma_i of the null, end exclusive
TRADING_DAYS = 252
MONTH_DAYS = 21

# Formation -------------------------------------------------------------------
L_GRID = range(1, 6)                        # years
GROUPINGS = ("G0", "G1", "G2")
MIN_SAMPLES = 5                             # OPTICS
METHODS = ("C1", "C2", "C3", "C3b", "C4")
BASKET = {"C2", "C3", "C3b"}
DAYS_PER_STOCK = 6                          # baskets need T / m >= 6
J0 = 5                                      # lags of the eigenanalysis

# Random walk null ------------------------------------------------------------
DRAWS = 1000
DRAWS_C1 = 10000
CHUNK = 20000                               # null spreads per measures call

# Trading ---------------------------------------------------------------------
H_GRID = (1, 3, 6, 12)                      # months
H_MAIN = 6                                  # Tables 2 and 3, Figure 1, Appendix B
ENTRY = 2.0                                 # |z| at entry
COST_BP = 10                                # per unit traded
BORROW = 0.0025                             # a year on the legs held short
EXITS = (0.5, 0.0, -0.5)                    # Appendix B, close at z n >= -exit
COSTS_BP = (5, 10, 25)
