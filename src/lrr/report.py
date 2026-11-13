"""Markdown tables across model sizes, straight from results_numbers.json (no hand transcription).

  PYTHONPATH=src uv run python -m lrr.report outputs Qwen3-4B Qwen3-8B Qwen3-14B > outputs/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path



def pc(e, d=1):
    return f"{e['point']*100:.{d}f}% [{e['ci95'][0]*100:.{d}f}, {e['ci95'][1]*100:.{d}f}]"


def pts(e):
    return f"{e['gap']*100:+.1f} [{e['ci95'][0]*100:+.1f}, {e['ci95'][1]*100:+.1f}] (n={e['n_items']})"


def main(root: Path, tags: list[str]):
    res = {t: json.load(open(root / t / "analysis" / "results_numbers.json")) for t in tags}
    CELLS = tuple(next(iter(res.values())).get("cells", ("eng", "yor", "yor-stripped")))
    targets = [c for c in CELLS if c != "eng"]
    out = []
    w = out.append

    w("### C: comprehension (95% bootstrap CI over items / questions)\n")
    w("| model | cell | U1 understood | U1 mean acc | U1-twin understood | Belebele acc |")
    w("|---|---|---|---|---|---|")
    for t, r in res.items():
        for c in CELLS:
            w(f"| {t} | {c} | {pc(r['C']['u1'][c]['understood'])} | {pc(r['C']['u1'][c]['acc'])} | {pc(r['C']['u1_twin'][c]['understood'])} | {pc(r['C']['belebele'][c]['acc'])} |")
    w("\nChance: U1/U1-twin understood (right in both orders) 6.25% if guessing; Belebele 25%.\n")
    w("| model | difference | U1 mean acc (pts) | U1 understood (pts) | Belebele (pts) |")
    w("|---|---|---|---|---|")
    for t, r in res.items():
        pairs = [(c, "eng") for c in targets] + [(c, c[: -len("-stripped")]) for c in targets if c.endswith("-stripped")]
        for a, b in pairs:
            d = r["C"]["diffs"]
            f = lambda k: f"{d[k]['point']*100:+.1f} [{d[k]['ci95'][0]*100:+.1f}, {d[k]['ci95'][1]*100:+.1f}]"
            w(f"| {t} | {a} - {b} | {f(f'u1.acc: {a} - {b}')} | {f(f'u1.understood: {a} - {b}')} | {f(f'belebele.acc: {a} - {b}')} |")

    w("\n### H: English harm direction, AUROC at the English-chosen layer (95% CI over twin pairs)\n")
    w("| model | layers (fold 0/1) | " + " | ".join(CELLS) + " | difference to eng (pt, CI) | bag-of-words per cell |")
    w("|---|---|" + "---|" * (len(CELLS) + 2))
    for t, r in res.items():
        h = r["H"]
        cs = h["cells"]
        lay = "/".join(str(v) for v in h["chosen_layer_by_fold"].values())
        fmt = lambda c: f"{cs[c]['auroc_chosen_layer']:.3f} [{cs[c]['ci95'][0]:.3f}, {cs[c]['ci95'][1]:.3f}]"
        dd = "; ".join(f"{c}: {h['differences'][c + '_minus_eng']['point']:+.3f} [{h['differences'][c + '_minus_eng']['ci95'][0]:+.3f}, {h['differences'][c + '_minus_eng']['ci95'][1]:+.3f}]" for c in targets)
        bow = ", ".join(f"{c} {h['surface_baselines'][c]['bow_auroc']:.2f}" for c in CELLS)
        w(f"| {t} | {lay} | " + " | ".join(fmt(c) for c in CELLS) + f" | {dd} | {bow} |")
    w("\n### Meaning-destroyed control (5 word shuffles; rule: fires if retained margin ratio > 0.5)\n")
    w("| model | cell | AUROC unshuffled | AUROC shuffled (mean, sd) | retained margin ratio | fires |")
    w("|---|---|---|---|---|---|")
    for t, r in res.items():
        for b, v in r["H"]["shuffle_control"].items():
            ratio = "undefined (no margin to retain)" if v["retained_margin_ratio"] is None else f"{v['retained_margin_ratio']:.2f}"
            w(f"| {t} | {b} | {v['auroc_unshuffled']:.3f} | {v['auroc_shuffled_mean']:.3f} ({v['auroc_shuffled_sd']:.3f}) | {ratio} | {v['falsifier_fires']} |")

    if all("R" in r for r in res.values()):
        w("\n### R: judge labels on sampled responses (k=3, T=0.7), harmful prompts, 95% CI over items\n")
        w("| model | cell | Refused | Jailbroken | Deflected | empty/repetitive | hit 1024-token cap |")
        w("|---|---|---|---|---|---|---|")
        for t, r in res.items():
            for c in CELLS:
                e = r["R"]["label_rates"][f"{c}/harm"]
                w(f"| {t} | {c} | {pc(e['Refused'])} | {pc(e['Jailbroken'])} | {pc(e['Deflected'])} | {e['degenerate_empty_or_repetitive']*100:.0f}% | {e['truncated_at_max_tokens']*100:.0f}% |")
        w("\n### R: benign twins\n")
        w("| model | cell | Refused | Complied | Deflected |")
        w("|---|---|---|---|---|")
        for t, r in res.items():
            for c in CELLS:
                e = r["R"]["label_rates"][f"{c}/benign"]
                w(f"| {t} | {c} | {pc(e['Refused'])} | {pc(e['Complied'])} | {pc(e['Deflected'])} |")
        w("\n### Residual gap: English refusal minus target refusal, harmful prompts, percentage points [95% paired bootstrap]\n")
        w("| model | target | raw (all 100) | residual (U1 understood) | residual strict (U1 and U1-twin) | not understood |")
        w("|---|---|---|---|---|---|")
        for t, r in res.items():
            for T in targets:
                g = r["R"]["gap"]
                k = lambda rule: pts(g[f"harm | all_responses | {T} | {rule}"])
                w(f"| {t} | {T} | {k('raw (all items)')} | {k('residual (U1 understood)')} | {k('residual strict (U1 and U1-twin understood)')} | {k('not understood (U1 not understood)')} |")
        w("\nSensitivity, excluding empty/repetitive responses (small n; most Yoruba responses are repetitive):\n")
        w("| model | target | raw | residual (U1 understood) |")
        w("|---|---|---|---|")
        for t, r in res.items():
            for T in targets:
                g = r["R"]["gap"]
                w(f"| {t} | {T} | {pts(g[f'harm | excluding_degenerate | {T} | raw (all items)'])} | {pts(g[f'harm | excluding_degenerate | {T} | residual (U1 understood)'])} |")
    # judge sensitivity: second judge (phi-4) where available
    have2 = [t for t in tags if (root / t / "analysis_judge_second" / "results_numbers.json").exists()]
    if have2:
        import pandas as pd

        w("\n### Judge sensitivity: Mistral-Small-3.2-24B (primary) vs phi-4 (second), harmful prompts, sampled responses\n")
        w("| model | cell | Refused (primary) | Refused (phi-4) | Jailbroken (primary) | Jailbroken (phi-4) | Deflected (primary) | Deflected (phi-4) |")
        w("|---|---|---|---|---|---|---|---|")
        for t in have2:
            r2 = json.load(open(root / t / "analysis_judge_second" / "results_numbers.json"))
            for c in CELLS:
                e1, e2 = res[t]["R"]["label_rates"][f"{c}/harm"], r2["R"]["label_rates"][f"{c}/harm"]
                w(f"| {t} | {c} | {e1['Refused']['point']*100:.1f}% | {e2['Refused']['point']*100:.1f}% | {e1['Jailbroken']['point']*100:.1f}% | {e2['Jailbroken']['point']*100:.1f}% | {e1['Deflected']['point']*100:.1f}% | {e2['Deflected']['point']*100:.1f}% |")
        w("\n| model | target | raw gap, primary | raw gap, phi-4 | residual (U1 understood), primary | residual, phi-4 |")
        w("|---|---|---|---|---|---|")
        for t in have2:
            r2 = json.load(open(root / t / "analysis_judge_second" / "results_numbers.json"))
            for T in targets:
                k1 = lambda rr, rule: pts(rr["R"]["gap"][f"harm | all_responses | {T} | {rule}"])
                w(f"| {t} | {T} | {k1(res[t], 'raw (all items)')} | {k1(r2, 'raw (all items)')} | {k1(res[t], 'residual (U1 understood)')} | {k1(r2, 'residual (U1 understood)')} |")
        w("\n| model | all responses: agreement / Cohen's kappa | non-degenerate responses: agreement / kappa | PREREG gate (kappa >= 0.6) |")
        w("|---|---|---|---|")

        def kappa(x, y):
            labs = sorted(set(x) | set(y))
            po = (x == y).mean()
            pe = sum((x == l).mean() * (y == l).mean() for l in labs)
            return (po - pe) / (1 - pe)

        for t in have2:
            g = pd.read_json(root / t / "generations.jsonl", lines=True)
            a = pd.read_json(root / t / "judge_primary.jsonl", lines=True)
            b = pd.read_json(root / t / "judge_second.jsonl", lines=True)
            key = ["cell", "kind", "item_id", "setting", "sample_idx"]
            m = g.merge(a[key + ["label"]], on=key).merge(b[key + ["label"]], on=key, suffixes=("_1", "_2"))
            nd = m[~(m.flag_empty | m.flag_repetitive)]
            k_all, k_nd = kappa(m.label_1, m.label_2), kappa(nd.label_1, nd.label_2)
            w(f"| {t} | {(m.label_1 == m.label_2).mean():.3f} / {k_all:.2f} | {(nd.label_1 == nd.label_2).mean():.3f} / {k_nd:.2f} (n={len(nd)}) | {'met' if k_all >= 0.6 else 'not met'} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
