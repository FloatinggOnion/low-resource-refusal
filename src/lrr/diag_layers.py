"""PREREG Addendum D1 (exploratory): transfer of the English harm direction by layer.

  PYTHONPATH=src uv run python -m lrr.diag_layers outputs outputs/diagnostics

Recomputes the H evaluation from the saved last-token states (outputs/<run>/ch/states/*.npy, private) and writes the
per-layer AUROC curves of the English difference-of-means direction in every cell (cross-fitted over the same two folds).
Only aggregate curves are written. A language shows transfer only if the cross-fold mean AUROC is >= 0.70 over >= 3
consecutive layers (PREREG D1); with 100 pairs the best of ~37 chance-level layers reaches about 0.60-0.65.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .analysis_h import evaluate
from .data import build_items

SIZES = ("4B", "8B", "14B")
SOURCES = (("", ("eng", "yor", "yor-stripped")), ("-swh-hau", ("swh", "hau", "hau-stripped")))
THRESH, RUN = 0.70, 3


def longest_run_at_least(curve, thresh: float = THRESH) -> int:
    best = cur = 0
    for v in curve:
        cur = cur + 1 if v >= thresh else 0
        best = max(best, cur)
    return best


def main(root: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    items = build_items()
    res = {}
    for size in SIZES:
        H = {}
        for suffix, cells in SOURCES:
            d = root / f"Qwen3-{size}{suffix}" / "ch" / "states"
            for c in cells:
                H[c] = {k: np.load(d / f"{c}_{k}.npy").astype(np.float32) for k in ("harm", "benign")}
        out, _ = evaluate(H, items, n_boot=200)
        res[size] = {"chosen_layer_by_fold": out["chosen_layer_by_fold"], "cells": {}}
        for c, v in out["cells"].items():
            cur = np.array(v["curve"])
            res[size]["cells"][c] = dict(curve=cur.tolist(), max=float(cur.max()), argmax=int(cur.argmax()),
                                         longest_run_ge_0_70=longest_run_at_least(cur), shows_transfer=bool(longest_run_at_least(cur) >= RUN),
                                         auroc_at_chosen_layer=v["auroc_chosen_layer"])
        print(size, {c: (round(v["max"], 2), v["longest_run_ge_0_70"]) for c, v in res[size]["cells"].items()})
    (outdir / "h_layers.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
