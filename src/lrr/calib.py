"""Hand-labelling sheet for judge calibration (private: contains harmful prompts and model output).

  PYTHONPATH=src uv run python -m lrr.calib outputs/Qwen3-4B [--judge primary] [--n 8]

Stratified sample over (cell, kind) of sampled-setting responses. The judge's label is withheld from the sheet
(kept in a separate key file) so the human label is not anchored. Fill `human_label` with
Refused / Jailbroken / Deflected (harmful) or Refused / Complied / Deflected (benign), then score with --score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--judge", default="primary")
    ap.add_argument("--n", type=int, default=8, help="responses per (cell, kind) stratum")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    out = d / "calibration"
    out.mkdir(exist_ok=True)
    if a.score:
        sheet = pd.read_csv(out / "sheet_filled.csv").merge(pd.read_csv(out / "key_judge_labels.csv"), on="sample_id")
        sheet = sheet[sheet.human_label.notna()]
        agree = (sheet.human_label.str.lower() == sheet.judge_label.str.lower())
        print(f"n={len(sheet)} agreement={agree.mean():.3f}")
        print(sheet.assign(agree=agree).groupby(["cell", "kind"]).agree.agg(["mean", "count"]))
        return
    r = load(d, a.judge)["r"]
    r = r[r.setting == "sampled"]
    s = r.groupby(["cell", "kind"]).sample(n=a.n, random_state=20261113)
    s = s.sample(frac=1, random_state=1).reset_index(drop=True)
    s["sample_id"] = range(len(s))
    s[["sample_id", "cell", "kind", "goal_en", "prompt", "response"]].assign(human_label="").to_csv(out / "sheet.csv", index=False)
    s[["sample_id", "cell", "kind", "item_id", "sample_idx"]].assign(judge_label=s["label"]).to_csv(out / "key_judge_labels.csv", index=False)
    print(f"wrote {out}/sheet.csv ({len(s)} rows) and key_judge_labels.csv")


if __name__ == "__main__":
    main()
