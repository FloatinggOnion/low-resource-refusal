"""Alignment checks on the pinned datasets. Downloads on first run."""

import pytest

from lrr.data import build_items, load_belebele, load_tuka
from lrr.text import count_marks, strip_diacritics, strip_hooked

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def items():
    return build_items()


def test_shape_and_categories(items):
    assert len(items) == 100
    assert items["item_id"].tolist() == list(range(100))
    counts = items["category"].value_counts()
    assert len(counts) == 10 and (counts == 10).all()


def test_twin_pairing(items):
    # Benign item i is the topic-matched twin of harmful item i: same JBB category and same JBB Index.
    assert (items["category"] == items["category_benign"]).all()
    assert (items["jbb_index_harm"] == items["jbb_index_benign"]).all()


def test_english_split_is_its_own_source(items):
    assert (items["harm_en"] == items["harm_eng_translation"]).all()
    assert (items["benign_en"] == items["benign_eng_translation"]).all()


def test_english_unchanged_by_stripping(items):
    for col in ("harm_en", "benign_en"):
        assert all(strip_diacritics(s) == s for s in items[col])


def test_every_yoruba_item_carries_marks_and_changes(items):
    for col in ("harm_yor", "benign_yor"):
        assert all(count_marks(s) > 0 for s in items[col])
        assert all(strip_diacritics(s) != s for s in items[col])
        assert all(len(strip_diacritics(s).split()) == len(s.split()) for s in items[col])


def test_no_duplicate_goals(items):
    assert items["harm_en"].is_unique and items["benign_en"].is_unique
    assert not set(items["harm_en"]) & set(items["benign_en"])


def test_swahili_and_hausa_columns_and_stripping(items):
    for lang in ("swh", "hau"):
        for kind in ("harm", "benign"):
            col = items[f"{kind}_{lang}"]
            assert len(col) == 100 and col.notna().all() and (col.str.len() > 0).all()
    hooked = set("ɓɗƙƴƁƊƘƳ")
    for kind in ("harm", "benign"):
        hau, swh = items[f"{kind}_hau"], items[f"{kind}_swh"]
        # Measured 2026-11-13: stripping changes 79% (harm) / 72% (benign) of Hausa prompts but merges only
        # 0.7% / 0.2% of word types: a much milder manipulation than Yoruba (about 10% of types).
        assert sum(strip_hooked(s) != s for s in hau) >= 70
        assert all(not any(c in hooked for c in strip_hooked(s)) for s in hau)
        assert all(len(strip_hooked(s).split()) == len(s.split()) for s in hau)
        assert all(strip_diacritics(s) == s and strip_hooked(s) == s for s in swh), "Swahili text has nothing to strip"


@pytest.mark.parametrize("lang", ["swh_Latn", "hau_Latn"])
def test_belebele_other_languages_align_with_english(lang):
    eng, other = load_belebele("eng_Latn"), load_belebele(lang)
    assert len(other) == 900
    assert (eng[["link", "question_number"]].values == other[["link", "question_number"]].values).all()
    assert (eng["correct_answer_num"].values == other["correct_answer_num"].values).all()


@pytest.mark.parametrize("lang", ["eng_Latn", "yor_Latn"])
def test_belebele_shape(lang):
    df = load_belebele(lang)
    assert len(df) == 900
    assert {"flores_passage", "question", "mc_answer1", "mc_answer4", "correct_answer_num"} <= set(df.columns)
    assert set(map(int, df["correct_answer_num"])) <= {1, 2, 3, 4}


def test_belebele_languages_align_row_by_row():
    eng, yor = load_belebele("eng_Latn"), load_belebele("yor_Latn")
    assert (eng[["link", "question_number"]].values == yor[["link", "question_number"]].values).all()
    assert (eng["correct_answer_num"].values == yor["correct_answer_num"].values).all()


def test_belebele_stripping_rates():
    """Stripping is applied to yor_Latn only. English Belebele is NOT strip-invariant
    (accented names/loanwords: 3.8% of passages, <0.5% of questions/answers), so
    `strip_diacritics` must never be run on the English cell. Measured 2026-11-13:
    yor changed: passages 99.8%, questions 97.6%, answers ~78%."""
    eng, yor = load_belebele("eng_Latn"), load_belebele("yor_Latn")
    changed = lambda s: (s.map(strip_diacritics) != s).mean()
    assert changed(eng["flores_passage"]) < 0.05
    for c in ("question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"):
        assert changed(eng[c]) < 0.01, c
    assert changed(yor["flores_passage"]) > 0.99
    assert changed(yor["question"]) > 0.95
    for c in ("mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"):
        assert changed(yor[c]) > 0.75, c
