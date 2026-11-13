"""Harm-direction analysis.

Direction = mean(harmful English) - mean(benign English) at every layer (Arditi et al. 2024 style),
fit on one stratified half of the twin pairs and evaluated on the other half, then folds swapped.
Stratified = split by JBB category; both members of a twin pair always sit in the same fold.
The layer is chosen on English training data only (inner 5-fold CV on the training fold), never
on Yoruba. AUROC is computed within each cell, so a language-wide offset in the hidden states
does not matter.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SEED = 20261113


# ----------------------------------------------------------------------------- primitives
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_layers(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.array([auroc(pos[:, l], neg[:, l]) for l in range(pos.shape[1])])


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    sp = np.sqrt((pos.var(0, ddof=1) + neg.var(0, ddof=1)) / 2)
    return (pos.mean(0) - neg.mean(0)) / np.maximum(sp, 1e-8)


def diff_dirs(h_harm: np.ndarray, h_benign: np.ndarray) -> np.ndarray:
    """(n,L,d),(n,L,d) -> unit directions (L,d)."""
    d = h_harm.mean(0) - h_benign.mean(0)
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)


def project(h: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    return np.einsum("nld,ld->nl", h, dirs)


def pair_accuracy(sh: np.ndarray, sb: np.ndarray) -> float:
    return float(np.mean((sh > sb) + 0.5 * (sh == sb)))


# ----------------------------------------------------------------------------- folds and layer choice
def make_folds(items: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """fold[item_id] in {0,1}: 5 of the 10 twin pairs of every category go to each fold."""
    rng = random.Random(f"{seed}|outer-folds")
    fold = np.zeros(len(items), dtype=int)
    for _, g in items.groupby("category"):
        ids = sorted(g["item_id"].tolist())
        rng.shuffle(ids)
        for k, i in enumerate(ids):
            fold[i] = 0 if k < len(ids) // 2 else 1
    return fold


def select_layer(h_harm, h_benign, cats, tag: str) -> tuple[int, np.ndarray, np.ndarray]:
    """English-only layer choice on a training fold: 5-fold inner CV, each inner fold holds out one
    twin pair per category. Highest mean held-out AUROC wins; ties (to 1e-9) go to the larger mean
    Cohen's d. Returns (layer, mean inner AUROC per layer, mean inner d per layer)."""
    cats = list(cats)
    rng = random.Random(tag)
    inner = np.zeros(len(cats), dtype=int)
    for cat in sorted(set(cats)):
        members = [k for k, c in enumerate(cats) if c == cat]
        rng.shuffle(members)
        for j, k in enumerate(members):
            inner[k] = j % 5
    aucs, ds = [], []
    for f in range(5):
        tr, te = inner != f, inner == f
        dirs = diff_dirs(h_harm[tr], h_benign[tr])
        sh, sb = project(h_harm[te], dirs), project(h_benign[te], dirs)
        aucs.append(auroc_layers(sh, sb))
        ds.append(cohens_d(sh, sb))
    auc, d = np.mean(aucs, 0), np.mean(ds, 0)
    best = int(np.lexsort((d, np.round(auc, 9)))[-1])
    return best, auc, d


# ----------------------------------------------------------------------------- main evaluation
def evaluate(H: dict[str, dict[str, np.ndarray]], items: pd.DataFrame, seed: int = SEED, n_boot: int = 2000) -> tuple[dict, pd.DataFrame]:
    """H[cell]['harm'|'benign'] -> (n_items, L+1, d). 'eng' must be present. Other cells (yor, yor-stripped,
    eng-shuf0.., yor-shuf0..) are evaluated with the English directions of the same fold."""
    folds = make_folds(items, seed)
    cats = items["category"].to_numpy()
    ids = items["item_id"].to_numpy()
    cells = list(H)
    n_layers = H["eng"]["harm"].shape[1]
    rng = np.random.default_rng(seed)

    fold_data = {c: [] for c in cells}
    chosen, inner_curves = {}, {}
    boot_idx = {}
    for f in (0, 1):
        tr, te = folds != f, folds == f
        layer, inner_auc, _ = select_layer(H["eng"]["harm"][tr], H["eng"]["benign"][tr], cats[tr], f"{seed}|inner|{f}")
        chosen[f], inner_curves[f] = layer, inner_auc
        dirs = diff_dirs(H["eng"]["harm"][tr], H["eng"]["benign"][tr])
        boot_idx[f] = rng.integers(0, te.sum(), size=(n_boot, te.sum()))  # shared across cells => paired CIs
        for c in cells:
            sh = project(H[c]["harm"][te], dirs)
            sb = project(H[c]["benign"][te], dirs)
            fold_data[c].append(dict(fold=f, layer=layer, curve=auroc_layers(sh, sb), sh=sh, sb=sb, ids=ids[te]))

    def cell_auc_boot(c):
        vals = np.zeros((n_boot, 2))
        for k, fd in enumerate(fold_data[c]):
            sh, sb = fd["sh"][:, fd["layer"]], fd["sb"][:, fd["layer"]]
            for b in range(n_boot):
                ix = boot_idx[fd["fold"]][b]
                vals[b, k] = auroc(sh[ix], sb[ix])
        return vals.mean(1)

    out = {"n_layers_incl_embedding": n_layers, "chosen_layer_by_fold": chosen, "cells": {}}
    boots = {}
    rows = []
    for c in cells:
        fds = fold_data[c]
        curve = np.mean([fd["curve"] for fd in fds], 0)
        at = float(np.mean([fd["curve"][fd["layer"]] for fd in fds]))
        pacc = float(np.mean([pair_accuracy(fd["sh"][:, fd["layer"]], fd["sb"][:, fd["layer"]]) for fd in fds]))
        boots[c] = cell_auc_boot(c)
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        out["cells"][c] = dict(auroc_chosen_layer=at, ci95=[float(lo), float(hi)], pair_accuracy=pacc, curve=curve.tolist(),
                               per_fold_auroc=[float(fd["curve"][fd["layer"]]) for fd in fds])
        for fd in fds:
            for k, i in enumerate(fd["ids"]):
                rows.append(dict(cell=c, item_id=int(i), fold=fd["fold"], layer=fd["layer"],
                                 score_harm=float(fd["sh"][k, fd["layer"]]), score_benign=float(fd["sb"][k, fd["layer"]])))
    out["inner_english_curve_by_fold"] = {f: v.tolist() for f, v in inner_curves.items()}

    # paired differences between cells (same bootstrap draws)
    out["differences"] = {}
    pairs = []
    for c in cells:
        if "-shuf" in c or c == "eng":
            continue
        pairs.append((c, "eng"))
        if c.endswith("-stripped"):
            pairs.append((c, c[: -len("-stripped")]))
    for a, b in pairs:
        if a in boots and b in boots:
            d = boots[a] - boots[b]
            out["differences"][f"{a}_minus_{b}"] = dict(
                point=out["cells"][a]["auroc_chosen_layer"] - out["cells"][b]["auroc_chosen_layer"],
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])

    # meaning-destroyed control, rule from PREREG.md item 5
    out["shuffle_control"] = {}
    for base in dict.fromkeys(c.split("-shuf")[0] for c in cells if "-shuf" in c):
        reps = [c for c in cells if c.startswith(f"{base}-shuf")]
        if not reps or base not in out["cells"]:
            continue
        a_orig = out["cells"][base]["auroc_chosen_layer"]
        a_shuf = [out["cells"][c]["auroc_chosen_layer"] for c in reps]
        margin = a_orig - 0.5
        ratios = [(a - 0.5) / margin for a in a_shuf] if margin > 0.05 else None
        out["shuffle_control"][base] = dict(
            auroc_unshuffled=a_orig, auroc_shuffled_mean=float(np.mean(a_shuf)), auroc_shuffled_sd=float(np.std(a_shuf)),
            auroc_shuffled_each=a_shuf,
            retained_margin_ratio=None if ratios is None else float(np.mean(ratios)),
            falsifier_fires=None if ratios is None else bool(np.mean(ratios) > 0.5),
            note="ratio=(AUC_shuf-0.5)/(AUC_orig-0.5); fires if > 0.5; undefined (None) if AUC_orig-0.5 <= 0.05")
    return out, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- surface-feature references
def surface_baselines(texts: dict[str, dict[str, list[str]]], token_counts: dict, items: pd.DataFrame, seed: int = SEED) -> dict:
    """Bag-of-words logistic regression trained and tested within each cell on the same two folds,
    plus the AUROC of prompt length alone. Both are references for what surface features can do."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    folds = make_folds(items, seed)
    out = {}
    for cell, t in texts.items():
        per_fold = []
        for f in (0, 1):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            x_tr = [t["harm"][i] for i in tr] + [t["benign"][i] for i in tr]
            y_tr = [1] * len(tr) + [0] * len(tr)
            x_te = [t["harm"][i] for i in te] + [t["benign"][i] for i in te]
            y_te = np.array([1] * len(te) + [0] * len(te))
            vec = CountVectorizer(binary=True, lowercase=True, token_pattern=r"[^\s.,;:!?\"'()]+")
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.fit_transform(x_tr), y_tr)
            s = clf.decision_function(vec.transform(x_te))
            per_fold.append(auroc(s[y_te == 1], s[y_te == 0]))
        tc = token_counts[cell]["content"]
        out[cell] = dict(bow_auroc=float(np.mean(per_fold)), bow_auroc_per_fold=per_fold,
                         length_auroc_harm_longer=auroc(tc["harm"].astype(float), tc["benign"].astype(float)))
    return out
