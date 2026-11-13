"""Modal entrypoints. Run from the repo root with PYTHONPATH=src, e.g.

  PYTHONPATH=src modal run --detach modal_app.py --model Qwen/Qwen3-4B --stage ch
  PYTHONPATH=src modal run --detach modal_app.py --model Qwen/Qwen3-4B --stage gen
  PYTHONPATH=src modal run --detach modal_app.py --model Qwen/Qwen3-4B --stage judge

Harmful prompts and generations live only inside the container and the private Volume `lrr-outputs`.
Fetch with:  modal volume get lrr-outputs /<tag> outputs/<tag>
"""

import json

import modal

app = modal.App("lrr-pilot")
out_vol = modal.Volume.from_name("lrr-outputs", create_if_missing=True)
hf_vol = modal.Volume.from_name("lrr-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.33.0", "scikit-learn==1.9.1", "pandas", "scipy", "huggingface_hub")
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which this slim image does not have
    # (first vLLM run died in the dummy sampler profile run). Use vLLM's native torch sampler instead.
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("lrr")
)
GPU = "H100"
JUDGE_DEFAULT = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
common = dict(image=image, gpu=GPU, volumes={"/out": out_vol, "/hf": hf_vol}, timeout=4 * 3600)


@app.function(**common)
def c_and_h(model_id: str, tag: str, belebele_n: int | None = None, cells: str = "eng,yor,yor-stripped") -> dict:
    from lrr.pipeline import run_c_h

    s = run_c_h(model_id, f"/out/{tag}/ch", device="cuda", dtype="bfloat16", belebele_n=belebele_n, cells=cells.split(","))
    out_vol.commit()
    hf_vol.commit()
    return s


@app.function(**common)
def generate(model_id: str, tag: str, max_new_tokens: int = 1024, cells: str = "eng,yor,yor-stripped", sampling: str = "design") -> dict:
    import os

    from lrr.gen import run_generation

    os.makedirs(f"/out/{tag}", exist_ok=True)
    s = run_generation(model_id, f"/out/{tag}/generations.jsonl", max_new_tokens=max_new_tokens, cells=tuple(cells.split(",")), sampling=sampling)
    out_vol.commit()
    hf_vol.commit()
    return s


@app.function(**common)
def judge(tag: str, judge_id: str, prompt_field: str = "goal_en", name: str = "primary") -> dict:
    from lrr.judge import run_judge

    out_vol.reload()
    s = run_judge(f"/out/{tag}/generations.jsonl", f"/out/{tag}/judge_{name}.jsonl", judge_id, prompt_field)
    out_vol.commit()
    hf_vol.commit()
    return s


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", stage: str = "ch", tag: str = "", belebele_n: int = 0,
         judge_id: str = JUDGE_DEFAULT, prompt_field: str = "goal_en", name: str = "primary",
         cells: str = "eng,yor,yor-stripped", sampling: str = "design"):
    tag = tag or model.split("/")[-1]
    if stage == "ch":
        s = c_and_h.remote(model, tag, belebele_n or None, cells)
        print(json.dumps({"c": s["c"], "h_cells": {k: {kk: v[kk] for kk in ("auroc_chosen_layer", "ci95", "pair_accuracy")} for k, v in s["h"]["cells"].items()},
                          "chosen_layer_by_fold": s["h"]["chosen_layer_by_fold"], "shuffle": s["h"]["shuffle_control"],
                          "surface": s["surface"], "seconds": s["meta"]["seconds"]}, indent=1, default=float))
    elif stage == "gen":
        print(json.dumps(generate.remote(model, tag, cells=cells, sampling=sampling), indent=1))
    elif stage == "judge":
        print(json.dumps(judge.remote(tag, judge_id, prompt_field, name), indent=1))
    else:
        raise SystemExit(f"unknown stage {stage}")
