"""Offline tests for the U1 item builder and the harm-direction analysis (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from lrr.analysis_h import auroc, evaluate, make_folds, select_layer
from lrr.probe import build_u1_specs, make_batches


@pytest.fixture(scope="module")
def items():
    cats = [f"cat{c}" for c in range(10)]
    return pd.DataFrame(
        dict(
            item_id=range(100),
            category=[cats[i // 10] for i in range(100)],
            harm_en=[f"harmful goal {i} about topic {i}" for i in range(100)],
            benign_en=[f"benign goal {i} about topic {i}" for i in range(100)],
        )
    )


def test_u1_specs_structure(items):
    specs = build_u1_specs(items)
    assert len(specs) == 2 * 100 * 2 * 2
    cat = dict(zip(items.item_id, items.category))
    for s in specs:
        i, kind = s["item_id"], s["kind"]
        own = items.loc[i, "harm_en" if kind == "harm" else "benign_en"]
        twin = items.loc[i, "benign_en" if kind == "harm" else "harm_en"]
        assert len(set(s["options"])) == 4
        assert s["options"][s["correct"]] == own
        if s["variant"] == "twin":
            assert s["options"].count(twin) == 1 and s["twin_text"] == twin
        else:
            assert twin not in s["options"] and s["twin_text"] is None
        # every other option is a same-category, same-split goal (twin excepted)
        for o in s["options"]:
            if o in (own, twin):
                continue
            j = int(o.split()[2])
            assert cat[j] == cat[i] and o.startswith(kind == "harm" and "harmful" or "benign")


def test_u1_orders_move_the_answer_and_are_balanced(items):
    specs = build_u1_specs(items)
    by = {}
    for s in specs:
        by.setdefault((s["kind"], s["item_id"], s["variant"]), {})[s["order"]] = s
    for pair in by.values():
        assert pair[0]["correct"] != pair[1]["correct"]
        assert pair[0]["options"] != pair[1]["options"]
    first = pd.Series([s["correct"] for s in specs if s["order"] == 0]).value_counts()
    assert first.min() >= 0.2 * first.sum()  # no letter gets starved


def test_u1_specs_reproducible(items):
    assert build_u1_specs(items) == build_u1_specs(items)


def test_make_batches_respects_budget():
    lens = [10, 500, 30, 200, 400, 5, 5, 5]
    for b in make_batches(lens, max_tokens=1000, max_bs=4):
        assert max(lens[i] for i in b) * len(b) <= 1000 or len(b) == 1
        assert len(b) <= 4
    assert sorted(i for b in make_batches(lens, 1000, 4) for i in b) == list(range(8))


def test_auroc_basics():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_folds_are_stratified_and_pair_complete(items):
    f = make_folds(items)
    for cat, g in items.groupby("category"):
        assert f[g.item_id.to_numpy()].sum() == len(g) // 2


def _synthetic_states(items, planted_layer=3, n_layers=6, d=32, strength=3.0, seed=0, direction=None):
    rng = np.random.default_rng(seed)
    n = len(items)
    u = np.zeros(d)
    u[0] = 1.0
    harm = rng.normal(size=(n, n_layers, d))
    ben = rng.normal(size=(n, n_layers, d))
    harm[:, planted_layer, :] += strength * u
    return harm.astype(np.float32), ben.astype(np.float32)


def test_evaluate_recovers_planted_direction_and_chance_controls(items):
    h_eng, b_eng = _synthetic_states(items, seed=1)
    h_yor, b_yor = _synthetic_states(items, seed=2, strength=1.0)  # weaker but same direction
    rng = np.random.default_rng(3)
    H = {
        "eng": {"harm": h_eng, "benign": b_eng},
        "yor": {"harm": h_yor, "benign": b_yor},
        # a 'shuffled' cell with no signal at all
        "eng-shuf0": {"harm": rng.normal(size=h_eng.shape).astype(np.float32), "benign": rng.normal(size=h_eng.shape).astype(np.float32)},
    }
    res, rows = evaluate(H, items, n_boot=200)
    assert set(res["chosen_layer_by_fold"].values()) == {3}  # English-only selection finds the planted layer
    assert res["cells"]["eng"]["auroc_chosen_layer"] > 0.95
    assert 0.6 < res["cells"]["yor"]["auroc_chosen_layer"] < res["cells"]["eng"]["auroc_chosen_layer"]
    assert abs(res["cells"]["eng-shuf0"]["auroc_chosen_layer"] - 0.5) < 0.1
    lo, hi = res["cells"]["eng"]["ci95"]
    assert lo <= res["cells"]["eng"]["auroc_chosen_layer"] <= hi
    assert len(rows) == 3 * 100  # every twin pair scored exactly once per cell (row holds harm and benign score)
    assert not rows.duplicated(["cell", "item_id"]).any()
    sc = res["shuffle_control"]["eng"]
    assert sc["falsifier_fires"] is False and sc["retained_margin_ratio"] < 0.2


def test_layer_selection_never_sees_yoruba(items):
    # select_layer takes only English arrays by construction; check it is deterministic for a tag
    h, b = _synthetic_states(items, seed=5)
    a = select_layer(h[:50], b[:50], items.category[:50], "t")
    c = select_layer(h[:50], b[:50], items.category[:50], "t")
    assert a[0] == c[0] == 3
