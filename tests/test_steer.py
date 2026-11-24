"""Hook mechanics of the steering code, checked on a tiny randomly initialised Qwen3 (CPU, no downloads).

HF's own output_hidden_states recorder runs before user hooks (it records a layer's output before our forward hook
changes it), so the checks observe what the NEXT layer actually receives, using pre-hooks registered after ours.
"""

import contextlib

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
    for i, h in seen.items():
        assert (h @ r).abs().max() < 1e-4, f"layer input {i}"
    assert (last_out @ r).abs().max() < 1e-4


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
