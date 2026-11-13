"""Text manipulations used in the pilot.

strip_diacritics  -> the `yor-stripped` cell (and Belebele yor_Latn stripped).
shuffle_words     -> the meaning-destroyed control for the harm-direction measure.
"""

from __future__ import annotations

import random
import unicodedata


def strip_diacritics(text: str) -> str:
    """Unicode NFD, drop every combining mark (category Mn), then NFC.

    For Yoruba this removes tone marks (acute, grave, macron) and under-dots
    (e with dot below -> e, o with dot below -> o, s with dot below -> s).
    Whitespace, punctuation and word order are untouched. English text that
    contains no combining marks is returned unchanged.

    Deliberately NOT handled: letters that are not decomposable marks, such as
    Hausa hooked letters (b/d/k/y with hook). The design maps those separately.
    """
    nfd = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", kept)


_HOOKED = str.maketrans(
    {"ɓ": "b", "Ɓ": "B", "ɗ": "d", "Ɗ": "D", "ƙ": "k", "Ƙ": "K", "ƴ": "y", "Ƴ": "Y"}
)


def strip_hooked(text: str) -> str:
    """Hausa `hau-stripped` cell: hooked letters to their plain Latin letters
    (b-hook -> b, d-hook -> d, k-hook -> k, y-hook -> y, and the capitals). Nothing else changes: no
    combining marks are removed, whitespace and word order are untouched, and text without these
    letters (English, Swahili) is returned unchanged. Merges real word pairs (e.g. kasa / k-hook-asa)."""
    return text.translate(_HOOKED)


STRIPPERS = {"yor": strip_diacritics, "hau": strip_hooked}


def count_marks(text: str) -> int:
    """Number of combining marks in the NFD form (descriptive statistic)."""
    return sum(1 for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) == "Mn")


def shuffle_words(text: str, seed: str) -> str:
    """Shuffle whitespace-separated words, deterministically given `seed`.

    Keeps the multiset of words (and so the tokens' surface content) but destroys
    word order. Re-draws up to 20 times so the result differs from the input
    whenever that is possible. Single-word or all-identical-word inputs are
    returned as they are. Output is joined with single spaces.
    """
    words = text.split()
    if len(words) < 2:
        return text
    rng = random.Random(seed)  # str seeds are hashed deterministically (not PYTHONHASHSEED)
    shuffled = words[:]
    for _ in range(20):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return " ".join(shuffled)
