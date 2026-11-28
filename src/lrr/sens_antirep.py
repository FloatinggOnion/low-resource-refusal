"""PREREG Addendum B: design sampling vs anti-repetition sampling (top-p 0.8, top-k 20, presence penalty 1.5).

  PYTHONPATH=src uv run python -m lrr.sens_antirep outputs Qwen3-4B Qwen3-8B Qwen3-14B > outputs/tables_antirep.md

Harmful prompts, sampled responses, primary judge. Design rows for eng/swh/hau come from <model>-swh-hau and for yor from
<model>; anti-repetition rows from <model>-antirep. The English cell is re-generated in each run, so each run's gap is
against its own English. 95% CIs: bootstrap over the 100 items (B=5000, seed 20261113), same draws for every cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cell", "kind", "item_id", "setting", "sample_idx"]
CELLS = ("eng", "yor", "swh", "hau")


def load(root: Path, tag: str) -> pd.DataFrame:
    g = pd.read_json(root / tag / "generations.jsonl", lines=True)
    j = pd.read_json(root / tag / "judge_primary.jsonl", lines=True)[KEY + ["label"]]
    m = g.merge(j, on=KEY, validate="1:1")
    m = m[(m.setting == "sampled") & (m.kind == "harm")].copy()
    m["deg"] = m.flag_empty | m.flag_repetitive
    return m


def per_item(m: pd.DataFrame, cell: str, f) -> np.ndarray:
    x = m[m.cell == cell].groupby("item_id").apply(f, include_groups=False)
    out = np.full(100, np.nan)
    out[x.index.to_numpy()] = x.to_numpy()
    return out


def main(root: Path, models: list[str]):
    rng = np.random.default_rng(20261113)
    idx = rng.integers(0, 100, size=(5000, 100))
    def ci(v, signed=False):
        lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5]) * 100
        if signed:
            return f"{v.mean()*100:+.1f} [{lo:+.1f}, {hi:+.1f}]"
        return f"{v.mean()*100:.1f}% [{lo:.1f}, {hi:.1f}]"

    print("### Decoding sensitivity (harmful prompts, sampled, primary judge)\n")
    print("Historical comparisons are paired by item only, without matched random draws. Each gap uses English from the same run.\n")
    print("| model | cell | setting | Refused | Jailbroken | Deflected | Non-empty/non-repetitive | Stopped | Both | Refusal gap to English (pts) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model in models:
        runs = (("design (Yoruba run)", model, ("eng", "yor")),
                ("design (Swahili/Hausa run)", f"{model}-swh-hau", ("eng", "swh", "hau")),
                ("anti-repetition", f"{model}-antirep", CELLS))
        for name, tag, cells in runs:
            m = load(root, tag)
            ref = per_item(m, "eng", lambda x: (x.label == "Refused").mean())
            for cell in cells:
                r = per_item(m, cell, lambda x: (x.label == "Refused").mean())
                j = per_item(m, cell, lambda x: (x.label == "Jailbroken").mean())
                d = per_item(m, cell, lambda x: (x.label == "Deflected").mean())
                gap = ref - r
                sub = m[m.cell == cell]
                clean = ~sub.deg
                stopped = sub.finish_reason.eq("stop")
                gp = ci(gap, signed=True) if cell != "eng" else "-"
                print(f"| {model} | {cell} | {name} | {ci(r)} | {ci(j)} | {ci(d)} | {clean.mean():.1%} | {stopped.mean():.1%} | {(clean & stopped).mean():.1%} | {gp} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
