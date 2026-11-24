"""PREREG Addendum C, checks C3 (language-modelling competence) and C4 (cross-lingual alignment by layer).

C3  bits per byte of Belebele passages (parallel across languages), plain text, no chat template. A document
    separator token is prepended so every real token is predicted.
C4  for each language and layer, retrieve each prompt's English counterpart among the 200 English prompts by cosine
    similarity of hidden states, after mean-centring each language's states. Two state definitions:
      pooled : mean over the prompt's content tokens (chat-templated input, template tokens excluded)
      last   : last token of the chat-templated prompt (the position used for H)
Only numbers are written (no prompt text), so the outputs are safe to keep alongside aggregate results.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import build_items, load_belebele
from .pipeline import load_model
from .probe import BELEBELE_CODE, cell_texts, chat_text, make_batches, parse_cell
from .text import STRIPPERS

CELLS = ("eng", "yor", "yor-stripped", "swh", "hau", "hau-stripped")


# ----------------------------------------------------------------------------- C3
@torch.no_grad()
def passage_nll(model, tok, texts: list[str], max_tokens: int = 8192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per text: total NLL in nats, UTF-8 byte count, token count (prepended separator excluded)."""
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    ids = [[sep] + tok(t, add_special_tokens=False).input_ids for t in texts]
    lens = [len(x) for x in ids]
    nll, nbytes, ntok = np.zeros(len(texts)), np.array([len(t.encode("utf-8")) for t in texts]), np.array(lens) - 1
    for idx in make_batches(lens, max_tokens, 32):
        T = max(lens[i] for i in idx)
        x = torch.full((len(idx), T), sep, dtype=torch.long)
        m = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, : lens[i]] = torch.tensor(ids[i])
            m[r, : lens[i]] = 1
        x, m = x.to(model.device), m.to(model.device)
        logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = x[:, 1:]
        ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * m[:, 1:]
        for r, i in enumerate(idx):
            nll[i] = -ll[r].sum().item()
    return nll, nbytes, ntok


def run_c3(model, tok, cells=CELLS) -> dict:
    out = {}
    cache = {}
    for cell in cells:
        lang, stripped = parse_cell(cell)
        code = BELEBELE_CODE[lang]
        if code not in cache:
            cache[code] = load_belebele(code)["flores_passage"].tolist()
        texts = cache[code]
        if stripped:
            texts = [STRIPPERS[lang](t) for t in texts]
        nll, nbytes, ntok = passage_nll(model, tok, texts)
        out[cell] = dict(nll_nats=nll.tolist(), bytes=nbytes.tolist(), tokens=ntok.tolist(),
                         bits_per_byte=float(nll.sum() / math.log(2) / nbytes.sum()), tokens_per_byte=float(ntok.sum() / nbytes.sum()))
    return out


# ----------------------------------------------------------------------------- C4
@torch.no_grad()
def prompt_states(model, tok, user_texts: list[str], max_tokens: int = 8192):
    """(pooled, last): each (n, L+1, d) float32. Right padding; content tokens found with offset mappings."""
    tok.padding_side = "right"
    texts, spans = [], []
    for u in user_texts:
        t = chat_text(tok, u)
        a = t.index(u)
        texts.append(t)
        spans.append((a, a + len(u)))
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    pooled = last = None
    for idx in make_batches(lens, max_tokens, 32):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
        off = enc.pop("offset_mapping")
        cm = torch.zeros(enc["input_ids"].shape, dtype=torch.bool)
        for r, i in enumerate(idx):
            a, b = spans[i]
            cm[r] = (off[r, :, 0] >= a) & (off[r, :, 1] <= b) & (off[r, :, 1] > off[r, :, 0])
        enc = enc.to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple of (b, T, d)
        cmd = cm.to(model.device).unsqueeze(-1)
        cnt = cmd.sum(1).clamp(min=1)
        lastpos = enc["attention_mask"].sum(1) - 1
        p = torch.stack([(h.float() * cmd).sum(1) / cnt for h in hs], dim=1).cpu().numpy()
        l = torch.stack([h[torch.arange(h.shape[0]), lastpos].float() for h in hs], dim=1).cpu().numpy()
        if pooled is None:
            pooled = np.zeros((len(texts),) + p.shape[1:], dtype=np.float32)
            last = np.zeros_like(pooled)
        pooled[idx], last[idx] = p, l
    return pooled, last


def retrieval(src: np.ndarray, eng: np.ndarray) -> dict:
    """src, eng: (200, L+1, d), same item order (100 harmful then 100 benign). Twin of i is (i+100) % 200."""
    n, nl, _ = src.shape
    top1, top5, twin = [], [], []
    twin_idx = (np.arange(n) + 100) % 200
    for l in range(nl):
        a = src[:, l] - src[:, l].mean(0)
        e = eng[:, l] - eng[:, l].mean(0)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
        s = a @ e.T
        order = np.argsort(-s, axis=1)
        pred = order[:, 0]
        ok = pred == np.arange(n)
        top1.append(float(ok.mean()))
        top5.append(float(np.mean([i in order[i, :5] for i in range(n)])))
        err = ~ok
        twin.append(float((pred[err] == twin_idx[err]).mean()) if err.any() else 0.0)
    return dict(top1=top1, top5=top5, twin_share_of_errors=twin)


def run_c4(model, tok, cells=CELLS) -> dict:
    items = build_items()
    states = {}
    for cell in cells:
        t = cell_texts(items, cell)
        states[cell] = prompt_states(model, tok, t["harm"] + t["benign"])
    out = {}
    for cell in cells:
        if cell == "eng":
            continue
        out[cell] = dict(pooled=retrieval(states[cell][0], states["eng"][0]), last=retrieval(states[cell][1], states["eng"][1]))
    return out


def run_lm_probe(model_id: str, out_dir: str, device: str = "cuda", dtype: str = "bfloat16", cells=CELLS) -> dict:
    t0 = time.time()
    model, tok = load_model(model_id, device, dtype)
    res = dict(model_id=model_id, cells=list(cells), c3=run_c3(model, tok, cells), c4=run_c4(model, tok, cells), seconds=time.time() - t0)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "lm_probe.json").write_text(json.dumps(res))
    return {k: v for k, v in res.items() if k not in ("c3",)} | {"c3_bits_per_byte": {c: v["bits_per_byte"] for c, v in res["c3"].items()}}
