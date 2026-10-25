"""For each sample, the groupings, the null cells and the evaluated spreads.

    python -m src.run
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one per process
    os.environ.setdefault(v, "1")

from src import evaluate, grouping, nulls  # noqa: E402
from src.config import RESULTS, SAMPLES  # noqa: E402

if __name__ == "__main__":
    for sample, span in SAMPLES.items():
        out = RESULTS / sample
        out.mkdir(parents=True, exist_ok=True)
        grouping.run(span, out / "groups.csv.gz")
        nulls.build(nulls.cells_needed(out / "groups.csv.gz"))
        evaluate.run(out / "groups.csv.gz", out / "cells.csv.gz")
