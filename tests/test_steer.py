"""Hook mechanics of the steering code, checked on a tiny randomly initialised Qwen3 (CPU, no downloads).

HF's own output_hidden_states recorder runs before user hooks (it records a layer's output before our forward hook
changes it), so the checks observe what the NEXT layer actually receives, using pre-hooks registered after ours.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
import torch

transformers = pytest.importorskip("transformers")

from lrr.steer import intervention  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).eval()


@contextlib.contextmanager
def capture_layer_inputs(model):
    seen = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def pre(_m, args, kwargs, i=i):
            seen[i] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    last = {}
    handles.append(model.model.layers[-1].register_forward_hook(lambda _m, _i, out: last.setdefault("out", (out[0] if isinstance(out, tuple) else out).detach().clone())))
    try:
        yield seen, last
    finally:
        for h in handles:
            h.remove()


def _run(model, x):
    with capture_layer_inputs(model) as (seen, last), torch.no_grad():
        logits = model(input_ids=x).logits
    return seen, last["out"], logits


def test_add_shifts_what_the_next_layer_receives_by_exactly_vec(tiny):
    x = torch.randint(0, 128, (2, 7))
    base, _, base_logits = _run(tiny, x)
    vec = torch.randn(32)
    with intervention(tiny, "add", layer=2, vec=vec):
        mod, _, mod_logits = _run(tiny, x)
    # layer=2 hooks decoder layer index 1; layer 2 (index) receives its output
    assert torch.allclose(mod[0], base[0], atol=1e-6) and torch.allclose(mod[1], base[1], atol=1e-6)
    assert torch.allclose(mod[2] - base[2], vec.expand_as(base[2]), atol=1e-5)
    assert (mod_logits - base_logits).abs().max() > 1e-3  # the model's output really changes


def test_ablation_removes_the_direction_from_every_layer_input_and_the_last_output(tiny):
    x = torch.randint(0, 128, (2, 7))
    d = torch.randn(32)
    r = d / d.norm()
    with intervention(tiny, "ablate", direction=d):
        seen, last_out, _ = _run(tiny, x)
        with torch.no_grad():
            final = tiny(x, output_hidden_states=True).hidden_states[-1]
    for i, h in seen.items():
        assert (h @ r).abs().max() < 1e-4, f"layer input {i}"
    assert (last_out @ r).abs().max() < 1e-4
    assert (final @ r).abs().max() < 1e-4


def test_final_index_adds_after_normalization(tiny):
    x = torch.randint(0, 128, (2, 7))
    vec = torch.randn(32)
    with torch.no_grad():
        base = tiny(x, output_hidden_states=True).hidden_states[-1]
        _, decoder_base, _ = _run(tiny, x)
        with intervention(tiny, "add", layer=len(tiny.model.layers), vec=vec):
            shifted = tiny(x, output_hidden_states=True).hidden_states[-1]
            _, decoder_shifted, _ = _run(tiny, x)
    torch.testing.assert_close(decoder_shifted, decoder_base)
    torch.testing.assert_close(shifted, base + vec)


def test_c7_uses_h_selected_layers(monkeypatch):
    from lrr import steer
    from lrr.analysis_h import evaluate

    rng = np.random.default_rng(0)
    eng = {kind: rng.normal(size=(100, 16, 12)) for kind in ("harm", "benign")}
    items = pd.DataFrame({"item_id": range(100), "category": np.repeat(range(10), 10),
                          "harm_en": ["harm"] * 100, "benign_en": ["benign"] * 100})
    monkeypatch.setattr(steer, "last_token_states", lambda _model, _tok, texts: eng[texts[0]])
    h, _ = evaluate({"eng": eng}, items, n_boot=10)
    expected = h["chosen_layer_by_fold"]
    _, selected = steer.fit_directions(None, None, items)
    assert {fold: layer for fold, (layer, _) in selected.items()} == expected
    _, frozen = steer.fit_directions(None, None, items, tuple(expected[f] for f in (0, 1)))
    for fold in expected:
        assert frozen[fold][0] == expected[fold]
        torch.testing.assert_close(frozen[fold][1], selected[fold][1])


def test_ablation_changes_the_output_and_leaves_orthogonal_directions_alone(tiny):
    x = torch.randint(0, 128, (1, 6))
    _, _, base_logits = _run(tiny, x)
    d = torch.randn(32)
    with intervention(tiny, "ablate", direction=d):
        _, _, mod_logits = _run(tiny, x)
    assert (mod_logits - base_logits).abs().max() > 1e-3


def test_hooks_are_removed_after_the_context(tiny):
    x = torch.randint(0, 128, (1, 5))
    _, _, before = _run(tiny, x)
    with intervention(tiny, "ablate", direction=torch.randn(32)):
        pass
    with intervention(tiny, "add", layer=1, vec=torch.randn(32)):
        pass
    _, _, after = _run(tiny, x)
    assert torch.allclose(before, after)


def test_unknown_mode_raises(tiny):
    with pytest.raises(ValueError):
        with intervention(tiny, "bogus"):
            pass


def test_sweep_layers_match_the_preregistered_indices():
    from lrr.steer import sweep_layers

    assert sweep_layers(36) == (9, 15, 21, 27, 33)
    assert sweep_layers(40) == (10, 17, 23, 30, 37)


def test_longest_run_rule_ignores_single_layer_peaks():
    from lrr.diag_layers import longest_run_at_least

    assert longest_run_at_least([0.5, 0.9, 0.5, 0.6]) == 1
    assert longest_run_at_least([0.5, 0.71, 0.8, 0.7, 0.4]) == 3


def test_replication_rule_needs_two_new_layers_and_an_adjacent_one():
    from lrr.diag_interv import replication_rule

    assert not replication_rule([])
    assert not replication_rule([20])          # a single adjacent hit is not enough
    assert not replication_rule([19, 23])      # two hits, none adjacent
    assert replication_rule([20, 23])
    assert replication_rule([19, 22])
