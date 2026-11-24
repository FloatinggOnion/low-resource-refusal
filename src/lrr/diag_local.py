"""PREREG Addendum C, checks C1 and C2: both run locally on data the pilot already produced.

  PYTHONPATH=src uv run python -m lrr.diag_local outputs outputs/diagnostics

C1 token fertility vs U1 success; C2 lexical-shortcut baseline for U1.
Bootstrap: B=2000, seed 20261113, resampling item ids (a harmful/benign twin pair always moves together).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import build_items
from .probe import build_u1_specs, cell_texts
from .text import strip_diacritics, strip_hooked

SEED = 20261113
B = 2000
SIZES = {"4B": ("Qwen3-4B", "Qwen3-4B-swh-hau"), "8B": ("Qwen3-8B", "Qwen3-8B-swh-hau"), "14B": ("Qwen3-14B", "Qwen3-14B-swh-hau")}
YOR_CELLS = ("yor", "yor-stripped")
TARGETS = ("yor", "yor-stripped", "swh", "hau", "hau-stripped")


# ----------------------------------------------------------------------------- C2 helpers
def fold(s: str) -> str:
    """Accent- and hook-folded lowercase, so the similarity is not penalised by orthography."""
    return strip_hooked(strip_diacritics(s)).lower()


def trigrams(s: str) -> set[str]:
    t = f" {fold(s)} "
    return {t[i : i + 3] for i in range(len(t) - 2)}


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def baseline_pick(prompt: str, options: list[str]) -> int:
    p = trigrams(prompt)
    sims = [jaccard(p, trigrams(o)) for o in options]
    return int(np.argmax(sims))


def phi(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(float), b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lexical_baseline(items: pd.DataFrame, cells=TARGETS) -> pd.DataFrame:
    """One row per (cell, kind, variant, item): did the trigram baseline pick the right English source?"""
    specs = [s for s in build_u1_specs(items) if s["order"] == 0]  # the pick does not depend on option order
    rows = []
    for cell in cells:
        texts = cell_texts(items, cell)
        for s in specs:
            pick = baseline_pick(texts[s["kind"]][s["item_id"]], s["options"])
            rows.append(dict(cell=cell, kind=s["kind"], variant=s["variant"], item_id=s["item_id"], baseline_correct=bool(pick == s["correct"])))
    return pd.DataFrame(rows)


def c2(root: Path, items: pd.DataFrame) -> dict:
    base = lexical_baseline(items)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 100, size=(B, 100))
    out = {}
    for size, (t_yor, t_sh) in SIZES.items():
        u1 = pd.concat([pd.read_csv(root / t_yor / "ch" / "u1_items.csv"), pd.read_csv(root / t_sh / "ch" / "u1_items.csv")])
        for cell in TARGETS:
            for variant in ("u1", "twin"):
                m = u1[(u1.cell == cell) & (u1.variant == variant)]
                b = base[(base.cell == cell) & (base.variant == variant)]
                # per item id (100): mean over the two kinds
                acc = m.groupby("item_id").acc.mean().reindex(range(100)).to_numpy()
                und = m.assign(u=(m.status == "understood")).groupby("item_id").u.mean().reindex(range(100)).to_numpy()
                bc = b.groupby("item_id").baseline_correct.mean().reindex(range(100)).to_numpy()
                wrong = bc < 0.5  # baseline wrong on (at least) one of the pair's two kinds -> treat as wrong
                def stat(ix):
                    w = wrong[ix]
                    return acc[ix].mean(), bc[ix].mean(), (acc[ix][w].mean() if w.any() else np.nan)
                boots = np.array([stat(ix) for ix in idx])
                point = stat(np.arange(100))
                ci = lambda j: [float(np.nanpercentile(boots[:, j], 2.5)), float(np.nanpercentile(boots[:, j], 97.5))]
                out[f"{size}|{cell}|{variant}"] = dict(
                    model_acc=dict(point=float(point[0]), ci95=ci(0)), baseline_acc=dict(point=float(point[1]), ci95=ci(1)),
                    model_acc_where_baseline_wrong=dict(point=None if np.isnan(point[2]) else float(point[2]), ci95=ci(2), n_items=int(wrong.sum())),
                    phi_baseline_vs_model_understood=phi(bc >= 0.5, und >= 0.5), gap_model_minus_baseline=float(point[0] - point[1]))
    return out


# ----------------------------------------------------------------------------- C1
def c1(root: Path, items: pd.DataFrame) -> dict:
    tc = {}
    for size, (t_yor, t_sh) in SIZES.items():
        a, b = pd.read_csv(root / t_yor / "ch" / "token_counts.csv"), pd.read_csv(root / t_sh / "ch" / "token_counts.csv")
        tc[size] = pd.concat([a, b[~b.cell.isin(["eng"])]])
    # one tokenizer for all sizes: check
    for size in ("8B", "14B"):
        assert (tc["4B"].sort_values(["cell", "kind", "item_id"]).content_tokens.to_numpy() == tc[size].sort_values(["cell", "kind", "item_id"]).content_tokens.to_numpy()).all()
    t = tc["4B"]
    eng = t[t.cell == "eng"].set_index(["kind", "item_id"]).content_tokens
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 100, size=(B, 100))
    out = {"fertility": {}, "coef": {}}
    for cell in TARGETS:
        texts = cell_texts(items, cell)
        words = {k: np.array([len(s.split()) for s in texts[k]]) for k in texts}
        tt = t[t.cell == cell].set_index(["kind", "item_id"]).content_tokens
        fert = (tt / eng.reindex(tt.index)).to_numpy()
        tpw = np.mean([tt.loc[("harm", i)] / words["harm"][i] for i in range(100)] + [tt.loc[("benign", i)] / words["benign"][i] for i in range(100)])
        out["fertility"][cell] = dict(mean_ratio_to_english=float(fert.mean()), mean_tokens_per_word=float(tpw), mean_tokens=float(tt.mean()))
    out["fertility"]["eng"] = dict(mean_ratio_to_english=1.0, mean_tokens_per_word=float(np.mean([eng.loc[("harm", i)] / len(items.harm_en[i].split()) for i in range(100)] + [eng.loc[("benign", i)] / len(items.benign_en[i].split()) for i in range(100)])), mean_tokens=float(eng.mean()))
    for size, (t_yor, t_sh) in SIZES.items():
        u1 = pd.concat([pd.read_csv(root / t_yor / "ch" / "u1_items.csv"), pd.read_csv(root / t_sh / "ch" / "u1_items.csv")])
        for cell in TARGETS:
            m = u1[(u1.cell == cell) & (u1.variant == "u1")].set_index(["kind", "item_id"]).status.eq("understood")
            tt = t[t.cell == cell].set_index(["kind", "item_id"]).content_tokens
            keys = [(k, i) for i in range(100) for k in ("harm", "benign")]
            y = np.array([m.loc[k] for k in keys], dtype=int)
            fert = np.array([tt.loc[k] / eng.loc[k] for k in keys])
            el = np.array([eng.loc[k] for k in keys], dtype=float)
            texts = cell_texts(items, cell)
            src = {"harm": items.harm_en.tolist(), "benign": items.benign_en.tolist()}
            lex = np.array([jaccard(trigrams(texts[k][i]), trigrams(src[k][i])) for (k, i) in keys])
            z = lambda v: (v - v.mean()) / v.std()
            X2 = np.column_stack([z(fert), z(el)])            # pre-registered model
            X3 = np.column_stack([z(fert), z(el), z(lex)])    # POST HOC robustness: adds lexical similarity to the English source
            def fit(X, rows, col=0):
                if y[rows].min() == y[rows].max():
                    return np.nan
                return LogisticRegression(C=np.inf, max_iter=2000).fit(X[rows], y[rows]).coef_[0][col]
            allrows = np.arange(len(y))
            entry = dict(share_understood=float(y.mean()), n=int(len(y)))
            for name, X, col in (("coef_per_sd_fertility", X2, 0), ("coef_fertility_with_lexsim", X3, 0), ("coef_lexsim", X3, 2)):
                boots = np.array([fit(X, np.array([2 * i + j for i in ix for j in (0, 1)]), col) for ix in idx])
                entry[name] = float(fit(X, allrows, col))
                entry[name + "_ci95"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
            entry["ci95"] = entry["coef_per_sd_fertility_ci95"]
            out["coef"][f"{size}|{cell}"] = entry
    return out


def c2b(root: Path, items: pd.DataFrame) -> dict:
    """Residual gap (harmful prompts, English refusal minus target refusal) restricted to items the model understood
    (U1, both orders) AND on which the trigram baseline is wrong: understanding that lexical matching cannot explain."""
    base = lexical_baseline(items)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 100, size=(B, 100))
    out = {}
    for size, (t_yor, t_sh) in SIZES.items():
        pi = pd.concat([pd.read_csv(root / t_yor / "analysis" / "per_item.csv"), pd.read_csv(root / t_sh / "analysis" / "per_item.csv")])
        eng = pi[(pi.cell == "eng") & (pi.kind == "harm")].drop_duplicates("item_id").set_index("item_id").sampled_r_Refused.reindex(range(100)).to_numpy()
        for cell in TARGETS:
            g = pi[(pi.cell == cell) & (pi.kind == "harm")].set_index("item_id")
            tgt = g.sampled_r_Refused.reindex(range(100)).to_numpy()
            und = (g.u1_status.reindex(range(100)) == "understood").to_numpy()
            bw = ~base[(base.cell == cell) & (base.variant == "u1") & (base.kind == "harm")].set_index("item_id").baseline_correct.reindex(range(100)).to_numpy().astype(bool)
            d = eng - tgt
            def est(mask, ix):
                m = mask[ix]
                return d[ix][m].mean() if m.any() else np.nan
            for name, mask in (("understood", und), ("understood_and_baseline_wrong", und & bw)):
                pt = est(mask, np.arange(100))
                bt = np.array([est(mask, ix) for ix in idx])
                out[f"{size}|{cell}|{name}"] = dict(gap=None if np.isnan(pt) else float(pt), ci95=[float(np.nanpercentile(bt, 2.5)), float(np.nanpercentile(bt, 97.5))] if mask.any() else None, n_items=int(mask.sum()))
    return out


def main(root: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    items = build_items()
    res = {"C1": c1(root, items), "C2": c2(root, items), "C2b": c2b(root, items)}
    (outdir / "c1_c2.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: len(v) for k, v in res.items()}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
