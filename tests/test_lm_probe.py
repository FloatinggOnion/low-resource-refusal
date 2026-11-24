import numpy as np

from lrr.lm_probe import retrieval


def _states(n=200, L=3, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, L, d)).astype(np.float32)


def test_retrieval_perfect_when_states_match_up_to_a_language_offset():
    eng = _states()
    src = eng + 5.0  # a constant per-language offset must not matter (mean-centring)
    r = retrieval(src, eng)
    assert all(x == 1.0 for x in r["top1"]) and all(x == 1.0 for x in r["top5"])


def test_retrieval_near_chance_for_unrelated_states():
    r = retrieval(_states(seed=1), _states(seed=2))
    assert max(r["top1"]) < 0.05  # chance is 0.5%


def test_twin_confusion_detected():
    eng = _states(seed=3)
    src = eng[(np.arange(200) + 100) % 200]  # every prompt is mapped onto its twin
    r = retrieval(src, eng)
    assert all(x == 0.0 for x in r["top1"]) and all(x == 1.0 for x in r["twin_share_of_errors"])
