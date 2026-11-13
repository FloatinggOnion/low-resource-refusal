"""C and H for one model: U1, U1-twin, Belebele (logits only) and last-token hidden states.

CLI (local smoke test):  uv run python -m lrr.pipeline --model Qwen/Qwen3-0.6B --device mps --dtype float32 --out outputs/smoke --belebele-n 60
On Modal the same run_c_h() is called from modal_app.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis_h import SEED, evaluate, surface_baselines
from .data import REVISIONS, build_items, load_belebele
from .hidden import last_token_states, token_counts
from .probe import BELEBELE_CODE, CELLS, build_u1_specs, cell_texts, parse_cell, run_belebele, run_u1, summarize_u1
from .text import shuffle_words


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_model(model_id: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    return model, tok


def h_texts(items: pd.DataFrame, n_shuffles: int, seed: int = SEED, cells=CELLS) -> dict[str, dict[str, list[str]]]:
    """Cell texts plus word-shuffled copies of every unstripped cell (eng and each language cell)."""
    texts = {c: cell_texts(items, c) for c in cells}
    for base in [c for c in cells if not c.endswith("-stripped")]:
        for r in range(n_shuffles):
            texts[f"{base}-shuf{r}"] = {
                kind: [shuffle_words(s, f"{seed}|{base}|{kind}|{i}|{r}") for i, s in enumerate(texts[base][kind])]
                for kind in ("harm", "benign")
            }
    return texts


def c_headlines(u1_item: pd.DataFrame, bel: pd.DataFrame) -> dict:
    out = {"u1": {}, "belebele": {}}
    for (cell, variant), g in u1_item.groupby(["cell", "variant"]):
        out["u1"][f"{cell}/{variant}"] = dict(
            understood=float((g.status == "understood").mean()), misread=float((g.status == "misread").mean()),
            uncertain=float((g.status == "uncertain").mean()), mean_acc=float(g.acc.mean()),
            picked_twin=None if g.picked_twin.isna().all() else float(g.picked_twin.mean()),
            probe_refusal_rate=float((g.p_mass < 0.5).mean()), n=int(len(g)))
    for cell, g in bel.groupby("cell"):
        out["belebele"][cell] = dict(acc=float(g.correct.mean()), n=int(len(g)), probe_refusal_rate=float((g.p_mass < 0.5).mean()))
    return out


def run_c_h(model_id: str, out_dir: str, device: str = "cuda", dtype: str = "bfloat16", belebele_n: int | None = None,
            n_shuffles: int = 5, save_states: bool = True, max_tokens: int = 16384, cells=CELLS) -> dict:
    cells = tuple(cells)
    assert cells[0] == "eng", "eng must come first: it fits the harm direction and is the reference cell"
    t0 = time.time()
    out = Path(out_dir)
    (out / "states").mkdir(parents=True, exist_ok=True)
    items = build_items()
    model, tok = load_model(model_id, device, dtype)
    t_load = time.time() - t0

    # ---- C
    specs = build_u1_specs(items)
    u1 = run_u1(model, tok, items, specs, max_tokens=max_tokens, cells=cells)
    u1.to_csv(out / "u1_scores.csv", index=False)
    u1_item = summarize_u1(u1)
    u1_item.to_csv(out / "u1_items.csv", index=False)
    codes = sorted({BELEBELE_CODE[parse_cell(c)[0]] for c in cells} | {"eng_Latn"})
    bel = run_belebele(model, tok, {code: load_belebele(code) for code in codes}, n=belebele_n, max_tokens=max_tokens, cells=cells)
    bel.to_csv(out / "belebele.csv", index=False)
    heads = c_headlines(u1_item, bel)
    t_c = time.time() - t0

    # ---- H
    texts = h_texts(items, n_shuffles, cells=cells)
    H, tc = {}, {}
    for cell, t in texts.items():
        H[cell] = {kind: last_token_states(model, tok, t[kind], max_tokens=min(max_tokens, 8192)) for kind in ("harm", "benign")}
        tc[cell] = {k: {kind: token_counts(tok, t[kind])[k] for kind in ("harm", "benign")} for k in ("content", "total")}
        if save_states:
            for kind in ("harm", "benign"):
                np.save(out / "states" / f"{cell}_{kind}.npy", H[cell][kind].astype(np.float16))
    h_res, h_items = evaluate(H, items)
    h_items.to_csv(out / "h_item_scores.csv", index=False)
    base = surface_baselines({c: texts[c] for c in cells}, tc, items)
    toks = {c: {k: {kind: dict(mean=float(v.mean()), median=float(np.median(v)), min=int(v.min()), max=int(v.max())) for kind, v in d.items()}
                for k, d in tc[c].items()} for c in cells}
    pd.DataFrame([dict(cell=c, kind=kind, item_id=i, content_tokens=int(tc[c]["content"][kind][i]), total_tokens=int(tc[c]["total"][kind][i]))
                  for c in cells for kind in ("harm", "benign") for i in range(len(items))]).to_csv(out / "token_counts.csv", index=False)

    import sklearn, transformers

    meta = dict(
        model_id=model_id, device=device, dtype=dtype, seed=SEED, belebele_n=belebele_n, n_shuffles=n_shuffles, cells=list(cells),
        revisions=REVISIONS, code_hash=code_hash(), enable_thinking=False,
        versions=dict(torch=str(torch.__version__), transformers=str(transformers.__version__), sklearn=str(sklearn.__version__), python=platform.python_version()),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seconds=dict(load=t_load, through_c=t_c, total=time.time() - t0),
        h_position="last token of apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
    )
    summary = dict(meta=meta, c=heads, h=h_res, surface=base, tokens=toks)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--belebele-n", type=int, default=None)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--cells", default=",".join(CELLS), help="comma-separated, eng first")
    a = ap.parse_args()
    s = run_c_h(a.model, a.out, a.device, a.dtype, a.belebele_n, a.shuffles, True, a.max_tokens, a.cells.split(","))
    print(json.dumps({"c": s["c"], "meta": s["meta"]["seconds"]}, indent=1, default=float))
