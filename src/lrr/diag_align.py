"""PREREG Addendum C4, last-token variant, from the hidden states the pilot already saved (no GPU).

  PYTHONPATH=src uv run python -m lrr.diag_align outputs outputs/diagnostics

Retrieval of each prompt's English counterpart among the 200 English prompts (100 harmful + 100 benign), cosine
similarity after mean-centring each language's states, per layer. The pooled-token variant needs a GPU run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .lm_probe import retrieval

SIZES = {"4B": ("Qwen3-4B", "Qwen3-4B-swh-hau"), "8B": ("Qwen3-8B", "Qwen3-8B-swh-hau"), "14B": ("Qwen3-14B", "Qwen3-14B-swh-hau")}
CELLS = ("yor", "yor-stripped", "swh", "hau", "hau-stripped")


def load_cell(root: Path, tag: str, cell: str) -> np.ndarray:
    d = root / tag / "ch" / "states"
    return np.concatenate([np.load(d / f"{cell}_harm.npy"), np.load(d / f"{cell}_benign.npy")]).astype(np.float32)


def main(root: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    for size, (t_yor, t_sh) in SIZES.items():
        eng = load_cell(root, t_yor, "eng")
        res[size] = {}
        for cell in CELLS:
            tag = t_yor if cell in ("yor", "yor-stripped") else t_sh
            res[size][cell] = retrieval(load_cell(root, tag, cell), eng)
    (outdir / "c4_last_token.json").write_text(json.dumps(res))
    for size, cells in res.items():
        for cell, r in cells.items():
            t1 = np.array(r["top1"])
            l = int(t1.argmax())
            print(f"{size:3s} {cell:13s} peak top-1 {t1.max()*100:5.1f}% at layer {l:2d}/{len(t1)-1}  final layer {t1[-1]*100:5.1f}%  top-5 at peak {r['top5'][l]*100:5.1f}%  twin share of errors at peak {r['twin_share_of_errors'][l]*100:4.0f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
