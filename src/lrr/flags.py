"""Cheap output-quality flags, kept next to (never merged into) the judge label."""

from __future__ import annotations

import re

_EN_STOP = set(
    "the of and to a in is that it for as with on be this are by or can not you your i we will from have an at "
    "if which but they their would should could may sorry cannot help".split()
)


def output_flags(text: str, finish_reason: str | None) -> dict:
    t = text.strip()
    words = t.split()
    empty = len(t) == 0
    repetitive = False
    if len(words) >= 40:
        grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        repetitive = len(set(grams)) / len(grams) < 0.5
    if not repetitive and len(words) >= 12:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        repetitive = len(lines) >= 6 and len(set(lines)) / len(lines) < 0.5
    toks = [w.lower().strip(".,;:!?\"'()[]") for w in words]
    en_share = (sum(w in _EN_STOP for w in toks) / len(toks)) if toks else 0.0
    return dict(
        flag_empty=empty,
        flag_repetitive=repetitive,
        flag_truncated=(finish_reason == "length"),
        english_stopword_share=round(en_share, 3),  # heuristic English-reply indicator, not a language-ID model
        n_words=len(words),
    )


def parse_judge(text: str, labels: tuple[str, ...]) -> tuple[str, str]:
    """Returns (label, reason). label is one of `labels` or 'Unparsed'."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            import json

            j = json.loads(m.group(0))
            lab = str(j.get("label", "")).strip()
            for L in labels:
                if lab.lower() == L.lower():
                    return L, str(j.get("reason", ""))
        except Exception:
            pass
    m2 = re.search(r'"label"\s*:\s*"([A-Za-z]+)"', text)
    if m2:
        for L in labels:
            if m2.group(1).lower() == L.lower():
                return L, ""
    return "Unparsed", text[:200]
