"""Cells beyond the original Yoruba pilot: Hausa stripping, cell parsing, texts, Belebele wiring."""

import pandas as pd
import pytest

from lrr.probe import BELEBELE_CODE, cell_texts, parse_cell, u1_user_text
from lrr.text import strip_diacritics, strip_hooked

B, D, K, Y = "ɓ", "ɗ", "ƙ", "ƴ"  # hooked b, d, k, y
BC, DC, KC, YC = "Ɓ", "Ɗ", "Ƙ", "Ƴ"  # capitals


def test_strip_hooked_maps_exactly_the_hooked_letters():
    s = f"{B}atanci {D}an {K}asa {Y}ar {BC}a {DC}a {KC}a {YC}a"
    assert strip_hooked(s) == "batanci dan kasa yar Ba Da Ka Ya"


def test_strip_hooked_leaves_everything_else_alone():
    for s in ["", "Andika makala ya kashfa ukidai kwamba rais", "Write a defamatory article", "a  b\tc\n", "da'awar cewa"]:
        assert strip_hooked(s) == s
    # combining marks are not touched (the design only maps hooked letters)
    assert strip_hooked("é") == "é"


def test_strip_hooked_same_words_same_order_idempotent():
    s = f"Rubuta labarin {B}atanci a kan da'awar cewa {K}asa {D}aya"
    out = strip_hooked(s)
    assert len(out.split()) == len(s.split())
    assert all(len(a) == len(b) for a, b in zip(out.split(), s.split()))  # one-to-one letter mapping
    assert strip_hooked(out) == out
    assert all(ord(c) < 128 for c in out)


def test_strip_hooked_merges_word_types():
    assert strip_hooked(f"{K}asa") == strip_hooked("kasa") == "kasa"


def test_stripping_rules_do_not_cross_languages():
    assert strip_diacritics(f"{K}asa") == f"{K}asa"  # Yoruba rule leaves Hausa hooks
    assert strip_hooked("ọ") == "ọ"  # Hausa rule leaves Yoruba under-dots


def test_parse_cell():
    assert parse_cell("eng") == ("eng", False)
    assert parse_cell("yor-stripped") == ("yor", True)
    assert parse_cell("hau-stripped") == ("hau", True)
    assert parse_cell("swh") == ("swh", False)
    for bad in ("swh-stripped", "eng-stripped", "xx", "hau-shuf0"):
        with pytest.raises(AssertionError):
            parse_cell(bad)


@pytest.fixture
def items():
    return pd.DataFrame(
        dict(
            item_id=[0, 1],
            harm_en=["h0", "h1"], benign_en=["b0", "b1"],
            harm_yor=["ọ", "é"], benign_yor=["x", "y"],
            harm_swh=["swh h0", "swh h1"], benign_swh=["swh b0", "swh b1"],
            harm_hau=[f"{K}a", "da"], benign_hau=[f"{B}a", "ba"],
        )
    )


def test_cell_texts_for_every_cell(items):
    assert cell_texts(items, "eng") == {"harm": ["h0", "h1"], "benign": ["b0", "b1"]}
    assert cell_texts(items, "swh")["harm"] == ["swh h0", "swh h1"]
    assert cell_texts(items, "hau")["harm"] == [f"{K}a", "da"]
    assert cell_texts(items, "hau-stripped") == {"harm": ["ka", "da"], "benign": ["ba", "ba"]}
    assert cell_texts(items, "yor-stripped")["harm"] == [strip_diacritics("ọ"), "e"]


def test_u1_prompt_names_the_language():
    assert "written in Hausa" in u1_user_text("t", "Hausa", ["a", "b", "c", "d"])


def test_belebele_codes_exist_for_every_language():
    assert set(BELEBELE_CODE) == {"eng", "yor", "swh", "hau"}
