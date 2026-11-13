"""EXPLORATORY (not in PREREG.md): in-language harm direction.

The pre-registered H uses the English direction. If that direction does not separate harmful from benign in
Yoruba, two things are possible: Yoruba has no harm structure at the model's last token, or it has one the English
direction does not align with. This fits the same difference-of-means direction *within* each cell (same folds, same
nested layer selection, but on that cell's own training fold) to tell them apart. Reuses analysis_h.evaluate by
passing the cell as the reference cell.

  PYTHONPATH=src uv run python -m lrr.h_inlang outputs/Qwen3-4B
Needs outputs/<tag>/ch/states/*.npy (downloaded from the Modal volume).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .analysis_h import evaluate
from .data import build_items


def run(d: Path) -> dict:
    items = build_items()
    st = d / "ch" / "states"
    load = lambda cell: {k: np.load(st / f"{cell}_{k}.npy").astype(np.float32) for k in ("harm", "benign")}
    out = {}
    for cell in ("eng", "yor", "yor-stripped"):
        H = {"eng": load(cell)}  # evaluate() treats the first cell as the direction-fitting cell
        if cell == "yor":
            for r in range(5):
                p = st / f"yor-shuf{r}_harm.npy"
                if p.exists():
                    H[f"eng-shuf{r}"] = load(f"yor-shuf{r}")
        res, _ = evaluate(H, items, n_boot=2000)
        out[cell] = dict(
            auroc=res["cells"]["eng"]["auroc_chosen_layer"], ci95=res["cells"]["eng"]["ci95"],
            pair_accuracy=res["cells"]["eng"]["pair_accuracy"], chosen_layer_by_fold=res["chosen_layer_by_fold"],
            shuffle=res["shuffle_control"].get("eng"),
        )
    return out


if __name__ == "__main__":
    d = Path(sys.argv[1])
    res = run(d)
    (d / "analysis").mkdir(exist_ok=True)
    (d / "analysis" / "h_in_language_exploratory.json").write_text(json.dumps(res, indent=1, default=float))
    for c, v in res.items():
        print(c, "AUROC %.3f [%.3f, %.3f]" % (v["auroc"], *v["ci95"]), "pair acc %.2f" % v["pair_accuracy"], "layers", v["chosen_layer_by_fold"],
              "| shuffled: " + (f"{v['shuffle']['auroc_shuffled_mean']:.3f} ratio {v['shuffle']['retained_margin_ratio']}" if v["shuffle"] else "n/a"))
