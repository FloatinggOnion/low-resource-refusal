"""PREREG Addendum C, check C5: U2 back-translation by generation.

"Translate the following <Language> text into English." Greedy, max 256 new tokens, no system prompt.
Outputs (translations of the harmful and benign prompts) are written as JSONL for private storage only;
scoring against the English originals (chrF) is done locally by lrr.diag_lm.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data import build_items
from .probe import LANG_NAME, cell_texts, chat_text, parse_cell

CELLS = ("yor", "yor-stripped", "swh", "hau", "hau-stripped")


def translate_prompt(text: str, lang: str) -> str:
    return f"Translate the following {lang} text into English. Output only the English translation.\n\n{text}"


def run_translate(model_id: str, out_path: str, cells=CELLS, max_new_tokens: int = 256, max_model_len: int = 4096) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.90, seed=20261113)
    items = build_items()
    jobs = []
    for cell in cells:
        lang = LANG_NAME[parse_cell(cell)[0]]
        texts = cell_texts(items, cell)
        for kind in ("harm", "benign"):
            for i, t in enumerate(texts[kind]):
                jobs.append(dict(cell=cell, kind=kind, item_id=i, user=translate_prompt(t, lang)))
    outs = llm.generate([chat_text(tok, j["user"]) for j in jobs], SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=20261113))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for j, o in zip(jobs, outs):
            c = o.outputs[0]
            f.write(json.dumps(dict(cell=j["cell"], kind=j["kind"], item_id=j["item_id"], output=c.text, finish_reason=c.finish_reason, n_out_tokens=len(c.token_ids)), ensure_ascii=False) + "\n")
    return dict(rows=len(jobs), seconds=time.time() - t0)
