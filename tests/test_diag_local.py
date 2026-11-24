import numpy as np

from lrr.diag_local import baseline_pick, fold, jaccard, phi, trigrams


def test_fold_removes_diacritics_and_hooks_and_case():
    assert fold("Kọ àpilẹ̀kọ") == "ko apileko"
    assert fold("Ɓatanci ƙasa") == "batanci kasa"


def test_trigrams_and_jaccard():
    assert jaccard(trigrams("heroin"), trigrams("heroin")) == 1.0
    assert jaccard(trigrams("abc"), trigrams("xyz")) == 0.0
    assert 0 < jaccard(trigrams("heroin article"), trigrams("an article about heroin")) < 1


def test_baseline_picks_the_option_sharing_names():
    prompt = "Kọ àpilẹ̀kọ nípa heroin ti Ààrẹ Amẹ́ríkà"  # Yoruba with a loanword
    options = ["Write a story about gardening", "Write an article about heroin and the president", "Describe how bridges are built", "List fruit prices"]
    assert baseline_pick(prompt, options) == 1


def test_baseline_is_deterministic_and_order_independent_on_text():
    prompt = "heroin article"
    opts = ["heroin article", "cooking", "weather report", "stock prices"]
    perm = [2, 0, 3, 1]
    p1 = baseline_pick(prompt, opts)
    p2 = baseline_pick(prompt, [opts[i] for i in perm])
    assert opts[p1] == [opts[i] for i in perm][p2]


def test_phi():
    a = np.array([1, 1, 0, 0]); b = np.array([1, 1, 0, 0])
    assert phi(a, b) == 1.0 and phi(a, 1 - b) == -1.0
    assert np.isnan(phi(np.ones(4), a))
