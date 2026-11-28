import json
import sys
from types import SimpleNamespace

import pandas as pd

from lrr import gen


def test_sample_seeds_are_invariant_to_cells_and_conditions(tmp_path, monkeypatch):
    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            assert all(p.n == 1 for p in params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=str(p.seed), finish_reason="stop", token_ids=[])])
                    for p in params]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: None)))
    monkeypatch.setattr(gen, "build_items", lambda: pd.DataFrame({"harm_en": ["h"], "benign_en": ["b"]}))
    monkeypatch.setattr(gen, "cell_texts", lambda _, cell: {"harm": [cell + "h"], "benign": [cell + "b"]})
    monkeypatch.setattr(gen, "chat_text", lambda _, prompt: prompt)

    def run(cells, sampling, suffix=""):
        path = tmp_path / "generations.jsonl"
        gen.run_generation("test-model", str(path), cells=cells, sampling=sampling, suffix=suffix)
        sampled = [row for row in map(json.loads, path.read_text().splitlines()) if row["setting"] == "sampled"]
        assert all(row["seed"] == int(row["response"]) for row in sampled)
        return {(row["cell"], row["kind"], row["item_id"], row["sample_idx"]): row["seed"] for row in sampled}

    baseline = run(("eng", "yor", "swh"), "design")
    assert len(baseline) == len(set(baseline.values())) == 18
    assert run(("swh", "eng", "yor"), "antirep") == baseline
    assert run(("swh", "eng"), "design_sampled", "Answer in English.") == {
        key: seed for key, seed in baseline.items() if key[0] != "yor"
    }
