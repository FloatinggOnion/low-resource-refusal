"""PREREG Addendum C7: refusal-direction ablation and steering (trigger stage).

Direction: the English difference-of-means vector (harmful minus benign English prompts, last token of the chat
template), at the layer chosen on English training data only, cross-fitted over the same two category-stratified
folds as H: items in fold f use the direction and layer fitted on the pairs NOT in fold f.

  (a) validation  : project the direction out after embeddings, each decoder layer and final normalization at all
                    positions on English harmful prompts. The registered rule requires a >= 30-point Refused drop.
  (b) steering    : add c x (raw difference-of-means vector) at the chosen hidden-state site at all positions, for
                    c in {0.5, 1, 1.5}, on Yoruba, Swahili and Hausa harmful prompts; equal-norm random-direction
                    controls. Greedy, one sample per prompt, 256 new tokens.
Responses are written as JSONL (private storage) in the same format the judge reads (setting = condition).
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .analysis_h import make_folds, select_layer
from .data import build_items
from .flags import output_flags
from .hidden import last_token_states
from .pipeline import load_model
from .probe import cell_texts, chat_text

SEED = 20261113
COEFS = (0.5, 1.0, 1.5)
CELLS = ("eng", "yor", "swh", "hau")


def _decoder_layers(model):
    return model.model.layers


def _apply(output, fn):
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


@contextlib.contextmanager
def intervention(model, mode: str, layer: int | None = None, vec: torch.Tensor | None = None, direction: torch.Tensor | None = None):
    """mode 'none' | 'add' | 'ablate'.
    add    : add vec at the corresponding HF hidden-state site, at every position.
    ablate : remove the direction after embeddings, decoder layers, and final norm."""
    handles = []
    if mode == "add":
        v = vec

        def add_hook(_m, _i, out):
            return _apply(out, lambda h: h + v.to(h.dtype).to(h.device))

        layers = _decoder_layers(model)
        if layer is None or not 0 <= layer <= len(layers):
            raise ValueError(f"invalid hidden-state index: {layer}")
        if layer == 0:
            site = model.model.embed_tokens
        elif layer == len(layers):
            site = model.model.norm
        else:
            site = layers[layer - 1]
        handles.append(site.register_forward_hook(add_hook))
    elif mode == "ablate":
        r = direction / direction.norm()

        def proj(h):
            rr = r.to(h.dtype).to(h.device)
            return h - (h @ rr).unsqueeze(-1) * rr

        handles.append(model.model.embed_tokens.register_forward_hook(lambda _m, _i, out: proj(out)))
        for layer_mod in _decoder_layers(model):
            handles.append(layer_mod.register_forward_hook(lambda _m, _i, out: _apply(out, proj)))
        handles.append(model.model.norm.register_forward_hook(lambda _m, _i, out: proj(out)))
    elif mode != "none":
        raise ValueError(mode)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate_batch(model, tok, user_texts: list[str], max_new_tokens: int = 256, bs: int = 48) -> list[tuple[str, str]]:
    """Greedy generation under whatever hooks are active. Returns (text, finish_reason)."""
    tok.padding_side = "left"
    texts = [chat_text(tok, u) for u in user_texts]
    order = np.argsort([-len(t) for t in texts])
    out: list[tuple[str, str] | None] = [None] * len(texts)
    for b in range(0, len(order), bs):
        idx = order[b : b + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1] :]
        for r, i in enumerate(idx):
            ids = new[r].tolist()
            stopped = tok.eos_token_id in ids or tok.convert_tokens_to_ids("<|im_end|>") in ids
            out[i] = (tok.decode(ids, skip_special_tokens=True), "stop" if stopped else "length")
    return out  # type: ignore[return-value]


def fit_directions(model, tok, items, h_layers: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Cross-fitted English directions. Returns (folds, {fold f: (layer, vec as float32 tensor)}), where the entry for
    f is fitted on the pairs NOT in fold f and is applied to items of fold f."""
    t = cell_texts(items, "eng")
    hh = last_token_states(model, tok, t["harm"])
    hb = last_token_states(model, tok, t["benign"])
    folds = make_folds(items)
    cats = items["category"].to_numpy()
    out = {}
    for f in (0, 1):
        tr = folds != f
        if h_layers is None:
            layer, _, _ = select_layer(hh[tr], hb[tr], cats[tr], f"{SEED}|inner|{f}")
        else:
            layer = h_layers[f]
        if not 0 <= layer < hh.shape[1]:
            raise ValueError(f"invalid H hidden-state index for fold {f}: {layer}")
        v = hh[tr][:, layer].mean(0) - hb[tr][:, layer].mean(0)
        out[f] = (int(layer), torch.tensor(v, dtype=torch.float32))
    return folds, out


def run_steer(model_id: str, out_path: str, cells=CELLS, coefs=COEFS, max_new_tokens: int = 256, device: str = "cuda", dtype: str = "bfloat16", kind: str = "harm",
              h_layers: tuple[int, int] | None = None) -> dict:
    """kind="harm": the pre-registered C7 design. kind="benign": specificity control (POST HOC, PREREG Addendum C changelog):
    the same vector added to the topic-matched benign twins, to tell a specific effect from a generic refusal switch."""
    t0 = time.time()
    model, tok = load_model(model_id, device, dtype)
    items = build_items()
    folds, dirs = fit_directions(model, tok, items, h_layers)
    rng = torch.Generator().manual_seed(SEED)
    rows = []

    def record(cell, condition, ids, texts, goals, results, meta):
        for i, (txt, fin), goal, u in zip(ids, results, goals, texts):
            r = dict(cell=cell, kind=kind, item_id=int(i), setting=condition, sample_idx=0, prompt=u, goal_en=goal, response=txt, finish_reason=fin, model_id=model_id, **meta)
            r.update(output_flags(txt, fin))
            rows.append(r)

    for cell in cells:
        texts = cell_texts(items, cell)[kind]
        goals = items["harm_en" if kind == "harm" else "benign_en"].tolist()
        for f in (0, 1):
            ids = [i for i in range(100) if folds[i] == f]
            layer, v = dirs[f]
            sub_t, sub_g = [texts[i] for i in ids], [goals[i] for i in ids]
            conds = [("baseline", "none", {})]
            if kind == "benign":
                cs = (1.5,) if cell == "eng" else (1.0, 1.5)
                for c in cs:
                    conds.append((f"add_c{c}", "add", dict(layer=layer, vec=c * v)))
                if cell != "eng":
                    rv = torch.randn(v.shape, generator=rng)
                    conds.append(("rand_c1.5", "add", dict(layer=layer, vec=rv / rv.norm() * (1.5 * v).norm())))
            elif cell == "eng":
                conds.append(("ablate", "ablate", dict(direction=v)))
            else:
                for c in coefs:
                    conds.append((f"add_c{c}", "add", dict(layer=layer, vec=c * v)))
                    rv = torch.randn(v.shape, generator=rng)
                    conds.append((f"rand_c{c}", "add", dict(layer=layer, vec=rv / rv.norm() * (c * v).norm())))
            for name, mode, kw in conds:
                with intervention(model, mode, **kw):
                    res = generate_batch(model, tok, sub_t, max_new_tokens)
                record(cell, name, ids, sub_t, sub_g, res, dict(fold=f, layer=layer, vec_norm=float(v.norm())))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(rows=len(rows), seconds=time.time() - t0, layers={f: dirs[f][0] for f in dirs})


SWEEP_C = 1.5
SWEEP_DEPTHS = (0.25, 0.42, 0.58, 0.75, 0.92)


def sweep_layers(n_decoder_layers: int) -> tuple[int, ...]:
    """PREREG Addendum D2: hidden-state indices at 25/42/58/75/92% of depth (36 layers -> 9,15,21,27,33; 40 -> 10,17,23,30,37)."""
    return tuple(int(round(n_decoder_layers * f)) for f in SWEEP_DEPTHS)


def run_steer_sweep(model_id: str, out_path: str, c: float = SWEEP_C, cells=("yor", "swh", "hau"), max_new_tokens: int = 256,
                    device: str = "cuda", dtype: str = "bfloat16") -> dict:
    """PREREG Addendum D2. For each layer in sweep_layers: the English difference-of-means vector at that layer (cross-fitted
    over the same two folds, no layer selection) is added at c x its raw size to English benign twins and, per cell, to
    harmful prompts and benign twins. Setting = "L{layer}_add_c{c}". Unsteered baselines are reused from the C7 runs."""
    t0 = time.time()
    model, tok = load_model(model_id, device, dtype)
    items = build_items()
    t = cell_texts(items, "eng")
    hh = last_token_states(model, tok, t["harm"])
    hb = last_token_states(model, tok, t["benign"])
    folds = make_folds(items)
    layers = sweep_layers(len(_decoder_layers(model)))
    rows = []
    for layer in layers:
        vecs = {f: torch.tensor(hh[folds != f][:, layer].mean(0) - hb[folds != f][:, layer].mean(0), dtype=torch.float32) for f in (0, 1)}
        for cell in ("eng",) + tuple(cells):
            for kind in (("benign",) if cell == "eng" else ("harm", "benign")):
                texts = cell_texts(items, cell)[kind]
                goals = items["harm_en" if kind == "harm" else "benign_en"].tolist()
                for f in (0, 1):
                    ids = [i for i in range(len(items)) if folds[i] == f]
                    sub_t = [texts[i] for i in ids]
                    with intervention(model, "add", layer=layer, vec=c * vecs[f]):
                        res = generate_batch(model, tok, sub_t, max_new_tokens, bs=len(sub_t))
                    for i, (txt, fin), u in zip(ids, res, sub_t):
                        r = dict(cell=cell, kind=kind, item_id=int(i), setting=f"L{layer}_add_c{c}", sample_idx=0, prompt=u, goal_en=goals[i],
                                 response=txt, finish_reason=fin, model_id=model_id, fold=f, layer=layer, vec_norm=float(vecs[f].norm()))
                        r.update(output_flags(txt, fin))
                        rows.append(r)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(rows=len(rows), seconds=time.time() - t0, layers=list(layers))


DENSE_LAYERS = (19, 20, 21, 22, 23)
DENSE_COEFS = (1.0, 1.5)


def run_steer_dense(model_id: str, out_path: str, layers=DENSE_LAYERS, coefs=DENSE_COEFS, rand_layer: int = 21, rand_c: float = 1.5,
                    cells=("yor", "swh", "hau"), max_new_tokens: int = 256, device: str = "cuda", dtype: str = "bfloat16") -> dict:
    """PREREG Addendum E: dense replication around one layer. Same construction as run_steer_sweep (English difference of means
    at each layer, cross-fitted, no layer selection) at several coefficients, plus an equal-norm random-direction control at
    one layer. Settings: "L{layer}_add_c{c}" and "L{layer}_rand_c{c}" (no English rows for the random control)."""
    t0 = time.time()
    model, tok = load_model(model_id, device, dtype)
    items = build_items()
    t = cell_texts(items, "eng")
    hh = last_token_states(model, tok, t["harm"])
    hb = last_token_states(model, tok, t["benign"])
    folds = make_folds(items)
    rng = torch.Generator().manual_seed(SEED)
    rows = []
    jobs = [(layer, c, "add") for c in coefs for layer in layers] + [(rand_layer, rand_c, "rand")]
    for layer, c, mode in jobs:
        vecs = {}
        for f in (0, 1):
            v = torch.tensor(hh[folds != f][:, layer].mean(0) - hb[folds != f][:, layer].mean(0), dtype=torch.float32)
            if mode == "rand":
                rv = torch.randn(v.shape, generator=rng)
                v = rv / rv.norm() * v.norm()
            vecs[f] = v
        for cell in (("eng",) if mode == "add" else ()) + tuple(cells):
            for kind in (("benign",) if cell == "eng" else ("harm", "benign")):
                texts = cell_texts(items, cell)[kind]
                goals = items["harm_en" if kind == "harm" else "benign_en"].tolist()
                for f in (0, 1):
                    ids = [i for i in range(len(items)) if folds[i] == f]
                    sub_t = [texts[i] for i in ids]
                    with intervention(model, "add", layer=layer, vec=c * vecs[f]):
                        res = generate_batch(model, tok, sub_t, max_new_tokens, bs=len(sub_t))
                    for i, (txt, fin), u in zip(ids, res, sub_t):
                        r = dict(cell=cell, kind=kind, item_id=int(i), setting=f"L{layer}_{mode}_c{c}", sample_idx=0, prompt=u, goal_en=goals[i],
                                 response=txt, finish_reason=fin, model_id=model_id, fold=f, layer=layer, vec_norm=float(vecs[f].norm()))
                        r.update(output_flags(txt, fin))
                        rows.append(r)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(rows=len(rows), seconds=time.time() - t0, jobs=[list(j) for j in jobs])
