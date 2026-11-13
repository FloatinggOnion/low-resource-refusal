"""R: generation with vLLM. Direct prompting: the user turn holds only the prompt, no system prompt,
no jailbreak template. Chat template applied exactly as in C and H (enable_thinking=False).

Two settings on the same prompts:
  sampled : k=3, temperature 0.7, top_p 0.9   (the design's setting, taken from Safety-IRT "Standard")
  greedy  : k=1, temperature 0                (TukaBench's setting, for comparability)
Raw responses are written as JSONL (private storage only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data import build_items
from .flags import output_flags
from .probe import CELLS, cell_texts, chat_text

SEED = 20261113


def run_generation(model_id: str, out_path: str, max_new_tokens: int = 1024, k: int = 3, max_model_len: int = 6144, cells=CELLS, sampling: str = "design") -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.90, seed=SEED)
    items = build_items()
    jobs = []
    for cell in cells:
        texts = cell_texts(items, cell)
        for kind in ("harm", "benign"):
            en = items["harm_en" if kind == "harm" else "benign_en"].tolist()
            for i, t in enumerate(texts[kind]):
                jobs.append(dict(cell=cell, kind=kind, item_id=i, prompt=t, goal_en=en[i]))
    prompts = [chat_text(tok, j["prompt"]) for j in jobs]

    if sampling == "design":
        settings = {
            "sampled": dict(n=k, temperature=0.7, top_p=0.9),
            "greedy": dict(n=1, temperature=0.0, top_p=1.0),
        }
    elif sampling == "antirep":
        # SENSITIVITY ONLY (PREREG Addendum B): tighter nucleus + top-k + presence penalty, to test whether the
        # repetitive Yoruba/Swahili/Hausa loops are a decoding artefact of the design setting.
        settings = {"sampled": dict(n=k, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)}
    else:
        raise ValueError(sampling)
    n_rows = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for name, kw in settings.items():
            seeds = [SEED + idx for idx in range(len(prompts))]
            sps = [SamplingParams(max_tokens=max_new_tokens, seed=s, **kw) for s in seeds]
            outs = llm.generate(prompts, sps)
            for j, s, o in zip(jobs, seeds, outs):
                for si, c in enumerate(o.outputs):
                    row = dict(**j, setting=name, sample_idx=si, response=c.text, finish_reason=c.finish_reason,
                               n_out_tokens=len(c.token_ids), seed=s, temperature=kw["temperature"], top_p=kw["top_p"], top_k=kw.get("top_k"), presence_penalty=kw.get("presence_penalty", 0.0), sampling=sampling,
                               max_new_tokens=max_new_tokens, model_id=model_id)
                    row.update(output_flags(c.text, c.finish_reason))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
    return dict(rows=n_rows, seconds=time.time() - t0, settings=settings, max_new_tokens=max_new_tokens)
