"""Data loading, pinned to dataset revisions (recorded 2026-11-13).

TukaBench  (CC-BY-NC 4.0): data/afri-jbb-{harm,benign}/{eng,yor,...}.csv  [Index, Goal, Goal_Translation]
JBB        (MIT)         : data/{harmful,benign}-behaviors.csv            [Index, Goal, Target, Behavior, Category, Source]
Belebele   (CC-BY-SA 4.0): data/{lang}.jsonl
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

REVISIONS = {
    "McGill-NLP/tukabench": "d9194efc2e0e5c8d7ea1b8c888df6016a0b32599",
    "JailbreakBench/JBB-Behaviors": "886acc352a31533ffbcf4ef22c744658688086fc",
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
}


def _file(repo: str, path: str) -> str:
    return hf_hub_download(repo, path, repo_type="dataset", revision=REVISIONS[repo])


def load_tuka(kind: str, lang: str) -> pd.DataFrame:
    """kind in {'harm','benign'}; lang e.g. 'eng','yor'. Columns: Index, Goal, Goal_Translation."""
    assert kind in {"harm", "benign"}
    return pd.read_csv(_file("McGill-NLP/tukabench", f"data/afri-jbb-{kind}/{lang}.csv"))


def load_jbb(kind: str) -> pd.DataFrame:
    """kind in {'harmful','benign'}."""
    assert kind in {"harmful", "benign"}
    return pd.read_csv(_file("JailbreakBench/JBB-Behaviors", f"data/{kind}-behaviors.csv"))


def load_belebele(lang: str) -> pd.DataFrame:
    """lang e.g. 'eng_Latn', 'yor_Latn'. 900 rows, sorted by (link, question_number).

    The files are NOT in the same row order across languages (only ~37% of rows
    share a gold answer position when paired by row number), so every language is
    sorted on the same key here and row i is then the same question everywhere.
    """
    df = pd.read_json(_file("facebook/belebele", f"data/{lang}.jsonl"), lines=True)
    df = df.sort_values(["link", "question_number"]).reset_index(drop=True)
    assert not df.duplicated(["link", "question_number"]).any()
    return df


LANGS = ("yor", "swh", "hau")


def build_items() -> pd.DataFrame:
    """One row per twin pair (100 rows).

    Columns: item_id, category, harm_en, benign_en, harm_<lang>, benign_<lang> for lang in LANGS
    (yor, swh, hau), harm_eng_translation, benign_eng_translation.
    item_id is TukaBench's Index; category comes from JBB, joined on the
    English goal text (not on Index) and checked for harmful/benign agreement.
    Every language split is checked to have the same Index and the same English Goal as the eng split.
    """
    out = {}
    tr = {}
    for kind in ("harm", "benign"):
        eng = load_tuka(kind, "eng").set_index("Index")
        out[kind] = (eng, None)
        for lang in LANGS:
            d = load_tuka(kind, lang).set_index("Index")
            assert list(eng.index) == list(d.index), f"{kind}: Index differs between eng and {lang}"
            assert (eng["Goal"] == d["Goal"]).all(), f"{kind}: English Goal differs between eng and {lang} splits"
            tr[(kind, lang)] = d["Goal_Translation"]

    jbb_h = load_jbb("harmful").set_index("Goal")
    jbb_b = load_jbb("benign").set_index("Goal")
    rows = []
    for idx in out["harm"][0].index:
        gh = out["harm"][0].loc[idx, "Goal"]
        gb = out["benign"][0].loc[idx, "Goal"]
        row = dict(
            item_id=int(idx),
            category=jbb_h.loc[gh, "Category"],
            category_benign=jbb_b.loc[gb, "Category"],
            jbb_index_harm=int(jbb_h.loc[gh, "Index"]),
            jbb_index_benign=int(jbb_b.loc[gb, "Index"]),
            harm_en=gh,
            benign_en=gb,
            harm_eng_translation=out["harm"][0].loc[idx, "Goal_Translation"],
            benign_eng_translation=out["benign"][0].loc[idx, "Goal_Translation"],
        )
        for lang in LANGS:
            row[f"harm_{lang}"] = tr[("harm", lang)].loc[idx]
            row[f"benign_{lang}"] = tr[("benign", lang)].loc[idx]
        rows.append(row)
    return pd.DataFrame(rows)
